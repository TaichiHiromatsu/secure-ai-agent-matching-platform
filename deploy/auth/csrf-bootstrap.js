(function secureMediationBrowserBootstrap() {
  'use strict';

  const bootstrapRequest = new XMLHttpRequest();
  bootstrapRequest.open('GET', '/auth/browser-bootstrap', false);
  bootstrapRequest.withCredentials = true;
  bootstrapRequest.send(null);
  if (bootstrapRequest.status !== 200) {
    window.location.replace('/login');
    throw new Error('Authenticated browser bootstrap failed');
  }

  const bootstrap = JSON.parse(bootstrapRequest.responseText);
  if (typeof bootstrap.subject !== 'string' || !bootstrap.subject ||
      typeof bootstrap.csrfToken !== 'string' || !bootstrap.csrfToken) {
    throw new Error('Authenticated browser bootstrap was invalid');
  }

  const subject = bootstrap.subject;
  const csrfToken = bootstrap.csrfToken;
  const unsafeMethods = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

  function sameOriginUrl(value) {
    const url = new URL(value, window.location.href);
    return url.origin === window.location.origin ? url : null;
  }

  function bindSessionPath(url) {
    url.pathname = url.pathname.replace(
      /^(\/apps\/payment_user_agent\/users\/)[^/]+(\/sessions(?:\/.*)?$)/,
      '$1' + encodeURIComponent(subject) + '$2'
    );
    return url;
  }

  function bindJsonBody(url, body) {
    if (url.pathname !== '/run' && url.pathname !== '/run_sse') return body;
    if (typeof body !== 'string') return body;
    try {
      const value = JSON.parse(body);
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        value.userId = subject;
        return JSON.stringify(value);
      }
    } catch (_) {
      // The server owns validation for malformed or non-JSON requests.
    }
    return body;
  }

  const nativeFetch = window.fetch.bind(window);
  window.fetch = function secureFetch(input, init) {
    const request = input instanceof Request ? input : null;
    const options = Object.assign({}, init || {});
    const method = String(options.method || (request && request.method) || 'GET').toUpperCase();
    const originalUrl = request ? request.url : String(input);
    const url = sameOriginUrl(originalUrl);
    if (!url) return nativeFetch(input, init);

    bindSessionPath(url);
    if (unsafeMethods.has(method)) {
      const headers = new Headers(options.headers || (request && request.headers) || undefined);
      headers.set('X-CSRF-Token', csrfToken);
      options.headers = headers;
      options.credentials = 'same-origin';
      options.body = bindJsonBody(url, options.body);
    }

    if (request) {
      return nativeFetch(new Request(url.toString(), request), options);
    }
    return nativeFetch(url.toString(), options);
  };

  const nativeOpen = XMLHttpRequest.prototype.open;
  const nativeSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function secureOpen(method, url) {
    const parsed = sameOriginUrl(String(url));
    this.__secureMediationMethod = String(method || 'GET').toUpperCase();
    this.__secureMediationUrl = parsed ? bindSessionPath(parsed) : null;
    const args = Array.prototype.slice.call(arguments);
    if (this.__secureMediationUrl) args[1] = this.__secureMediationUrl.toString();
    return nativeOpen.apply(this, args);
  };
  XMLHttpRequest.prototype.send = function secureSend(body) {
    if (this.__secureMediationUrl && unsafeMethods.has(this.__secureMediationMethod)) {
      this.setRequestHeader('X-CSRF-Token', csrfToken);
      body = bindJsonBody(this.__secureMediationUrl, body);
    }
    return nativeSend.call(this, body);
  };

  window.__secureMediationBrowserReady = {subject: subject};
})();
