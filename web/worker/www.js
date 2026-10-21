// www.epzero.dev: a permanent redirect to the same path on epzero.dev, so the site has one address.
export default {
  fetch(request) {
    const url = new URL(request.url);
    url.hostname = "epzero.dev";
    url.protocol = "https:";
    url.port = "";
    return Response.redirect(url.toString(), 301);
  },
};
