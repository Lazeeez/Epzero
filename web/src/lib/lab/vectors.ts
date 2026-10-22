// Reads a file of vectors in the formats people keep embeddings in: NumPy .npy (float32/64/16,
// int8, uint8), the TEXMEX .fvecs/.bvecs of ann-benchmarks, CSV/TSV/plain text, and JSON (an
// array of arrays, or of objects with an embedding) or JSON Lines. Everything becomes float32.

export interface Vectors {
  data: Float32Array;
  n: number;
  dim: number;
  /** What was read, for the summary ("NumPy float32", "CSV"...). */
  format: string;
}

export class VectorFileError extends Error {}

/** Most values (n x dim) the lab takes: the index keeps its own copy, inside 4 GB of memory. */
export const MAX_VALUES = 150_000_000;

function check(n: number, dim: number) {
  if (n < 2 || dim < 1) throw new VectorFileError("The file needs at least two vectors.");
  if (dim > 4096) throw new VectorFileError(`${dim} dimensions: the lab takes up to 4,096.`);
  if (n * dim > MAX_VALUES)
    throw new VectorFileError(
      `${n.toLocaleString()} × ${dim.toLocaleString()} values is more than this browser lab holds (${(MAX_VALUES / 1e6).toFixed(0)} million: e.g. 1M × 128 or 100k × 1,536). Try a sample.`,
    );
}

const half = (h: number) => {
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  const s = h & 0x8000 ? -1 : 1;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
};

async function readNpy(file: File): Promise<Vectors> {
  const head = new Uint8Array(await file.slice(0, 12).arrayBuffer());
  if (head[0] !== 0x93 || String.fromCharCode(...head.slice(1, 6)) !== "NUMPY") throw new VectorFileError("Not a NumPy .npy file.");
  const major = head[6];
  const view = new DataView(head.buffer);
  const headerLength = major >= 2 ? view.getUint32(8, true) : view.getUint16(8, true);
  const offset = (major >= 2 ? 12 : 10) + headerLength;
  const header = new TextDecoder().decode(await file.slice(major >= 2 ? 12 : 10, offset).arrayBuffer());
  const descr = /'descr':\s*'([<>|=]?)([fiu])(\d+)'/.exec(header);
  const shape = /'shape':\s*\((\d+),\s*(\d+)\s*,?\)/.exec(header);
  if (/'fortran_order':\s*True/.test(header)) throw new VectorFileError("This .npy is in Fortran order; save it with np.ascontiguousarray first.");
  if (!descr || !shape) throw new VectorFileError("Expected a 2-D NumPy array (vectors × dimensions) of numbers.");
  if (descr[1] === ">") throw new VectorFileError("Big-endian .npy files aren't supported.");
  const [n, dim] = [Number(shape[1]), Number(shape[2])];
  check(n, dim);
  const kind = descr[2] + descr[3];
  const bytes = await file.slice(offset).arrayBuffer();
  const count = n * dim;
  let data: Float32Array;
  if (kind === "f4") data = new Float32Array(bytes, 0, count);
  else if (kind === "f8") data = Float32Array.from(new Float64Array(bytes, 0, count));
  else if (kind === "f2") data = Float32Array.from(new Uint16Array(bytes, 0, count), half);
  else if (kind === "i1") data = Float32Array.from(new Int8Array(bytes, 0, count));
  else if (kind === "u1") data = Float32Array.from(new Uint8Array(bytes, 0, count));
  else if (kind === "i4") data = Float32Array.from(new Int32Array(bytes, 0, count));
  else throw new VectorFileError(`NumPy type ${kind} isn't supported: use float32.`);
  const names: Record<string, string> = { f4: "float32", f8: "float64", f2: "float16", i1: "int8", u1: "uint8", i4: "int32" };
  return { data, n, dim, format: `NumPy ${names[kind]}` };
}

async function readVecs(file: File, bytesPerValue: 1 | 4): Promise<Vectors> {
  const buffer = await file.arrayBuffer();
  const view = new DataView(buffer);
  if (buffer.byteLength < 4) throw new VectorFileError("The file is empty.");
  const dim = view.getInt32(0, true);
  const row = 4 + dim * bytesPerValue;
  if (dim <= 0 || buffer.byteLength % row !== 0) throw new VectorFileError("Not a valid .fvecs/.bvecs file (each row: an int32 dimension, then its values).");
  const n = buffer.byteLength / row;
  check(n, dim);
  const data = new Float32Array(n * dim);
  for (let i = 0; i < n; i++) {
    if (view.getInt32(i * row, true) !== dim) throw new VectorFileError(`Row ${i + 1} has a different dimension.`);
    if (bytesPerValue === 4) data.set(new Float32Array(buffer.slice(i * row + 4, (i + 1) * row)), i * dim);
    else data.set(new Uint8Array(buffer, i * row + 4, dim), i * dim);
  }
  return { data, n, dim, format: bytesPerValue === 4 ? ".fvecs (float32)" : ".bvecs (uint8)" };
}

// Collects rows of numbers into a growing float32 buffer.
class Rows {
  data = new Float32Array(1 << 16);
  n = 0;
  dim = 0;
  push(values: ArrayLike<number>, where: string) {
    if (this.dim === 0) this.dim = values.length;
    if (values.length !== this.dim) throw new VectorFileError(`${where} has ${values.length} values, the first row ${this.dim}.`);
    if ((this.n + 1) * this.dim > MAX_VALUES) check(this.n + 1, this.dim);
    if ((this.n + 1) * this.dim > this.data.length) {
      const grown = new Float32Array(Math.max(this.data.length * 2, (this.n + 1) * this.dim));
      grown.set(this.data);
      this.data = grown;
    }
    for (let j = 0; j < this.dim; j++) {
      const x = values[j];
      if (!Number.isFinite(x)) throw new VectorFileError(`${where} has a value that isn't a number.`);
      this.data[this.n * this.dim + j] = x;
    }
    this.n++;
  }
  done(format: string): Vectors {
    check(this.n, this.dim);
    return { data: this.data.slice(0, this.n * this.dim), n: this.n, dim: this.dim, format };
  }
}

// Text, a line at a time without holding the whole file as one string.
async function* lines(file: File): AsyncGenerator<string> {
  const reader = file.stream().pipeThrough(new TextDecoderStream()).getReader();
  let rest = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    rest += value;
    let start = 0;
    for (let end = rest.indexOf("\n"); end !== -1; end = rest.indexOf("\n", start)) {
      yield rest.slice(start, end);
      start = end + 1;
    }
    rest = rest.slice(start);
  }
  if (rest) yield rest;
}

async function readText(file: File): Promise<Vectors> {
  const rows = new Rows();
  let separator: RegExp | null = null;
  let line = 0;
  let skipped = false;
  for await (const raw of lines(file)) {
    line++;
    const text = raw.trim().replace(/^\[|\]$/g, "");
    if (!text) continue;
    separator ??= text.includes(",") ? /\s*,\s*/ : text.includes("\t") ? /\t/ : text.includes(";") ? /\s*;\s*/ : /\s+/;
    const cells = text.split(separator);
    const values = cells.map(Number);
    if (rows.n === 0 && !skipped && values.some((v, i) => !Number.isFinite(v) || cells[i] === "")) {
      skipped = true; // a header row
      continue;
    }
    rows.push(values, `Line ${line}`);
  }
  return rows.done(file.name.toLowerCase().endsWith(".tsv") || separator?.source === "\\t" ? "TSV" : "CSV");
}

// The vector in a JSON value: an array of numbers, or the first such array among an object's fields.
function vectorIn(value: unknown): number[] | null {
  if (Array.isArray(value) && value.length && typeof value[0] === "number") return value as number[];
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const object = value as Record<string, unknown>;
    for (const key of ["embedding", "vector", "values", "emb"]) if (Array.isArray(object[key])) return object[key] as number[];
    for (const field of Object.values(object)) if (Array.isArray(field) && typeof field[0] === "number") return field as number[];
  }
  return null;
}

async function readJson(file: File): Promise<Vectors> {
  if (file.size > 400e6) throw new VectorFileError("JSON this large won't parse in a browser; use .npy, or JSON Lines.");
  let parsed: unknown;
  try {
    parsed = JSON.parse(await file.text());
  } catch {
    throw new VectorFileError("This isn't valid JSON.");
  }
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    const object = parsed as Record<string, unknown>;
    parsed = object.vectors ?? object.embeddings ?? object.data ?? Object.values(object).find(Array.isArray);
  }
  if (!Array.isArray(parsed)) throw new VectorFileError("Expected an array of vectors.");
  const rows = new Rows();
  parsed.forEach((item, i) => {
    const vector = vectorIn(item);
    if (!vector) throw new VectorFileError(`Item ${i + 1} isn't a vector.`);
    rows.push(vector, `Item ${i + 1}`);
  });
  return rows.done("JSON");
}

async function readJsonLines(file: File): Promise<Vectors> {
  const rows = new Rows();
  let line = 0;
  for await (const raw of lines(file)) {
    line++;
    if (!raw.trim()) continue;
    let value: unknown;
    try {
      value = JSON.parse(raw);
    } catch {
      throw new VectorFileError(`Line ${line} isn't valid JSON.`);
    }
    const vector = vectorIn(value);
    if (!vector) throw new VectorFileError(`Line ${line} has no vector.`);
    rows.push(vector, `Line ${line}`);
  }
  return rows.done("JSON Lines");
}

export async function readVectors(file: File): Promise<Vectors> {
  const name = file.name.toLowerCase();
  const magic = new Uint8Array(await file.slice(0, 6).arrayBuffer());
  if (name.endsWith(".npy") || (magic[0] === 0x93 && String.fromCharCode(...magic.slice(1, 6)) === "NUMPY")) return readNpy(file);
  if (name.endsWith(".fvecs")) return readVecs(file, 4);
  if (name.endsWith(".bvecs")) return readVecs(file, 1);
  if (name.endsWith(".jsonl") || name.endsWith(".ndjson")) return readJsonLines(file);
  if (name.endsWith(".json")) return readJson(file);
  if (name.endsWith(".npz") || name.endsWith(".parquet") || name.endsWith(".h5") || name.endsWith(".hdf5"))
    throw new VectorFileError("That format needs a library to unpack: save the array with np.save(\"vectors.npy\", x) instead.");
  return readText(file);
}

/** Mean vector length, and whether every vector is (nearly) unit length: cosine data usually is. */
export function norms(v: Vectors): { mean: number; unit: boolean } {
  let sum = 0;
  let unit = true;
  const step = Math.max(1, Math.floor(v.n / 2000)); // a sample is plenty
  let seen = 0;
  for (let i = 0; i < v.n; i += step) {
    let s = 0;
    for (let j = 0; j < v.dim; j++) s += v.data[i * v.dim + j] ** 2;
    const norm = Math.sqrt(s);
    sum += norm;
    seen++;
    if (Math.abs(norm - 1) > 0.02) unit = false;
  }
  return { mean: sum / seen, unit };
}
