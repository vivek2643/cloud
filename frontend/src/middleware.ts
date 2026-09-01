import { NextResponse, type NextRequest } from "next/server";

export const INVITE_COOKIE = "edso_invite";

/**
 * Front-door gate for the private beta.
 *
 * Only /signup is gated. Deliberately NOT gated:
 *   /login          existing users must always be able to get back in
 *   /u/[token]      upload links are used by people with no account at all
 *   /               the landing page stays public
 *   /drive/*        already behind real auth
 *
 * This is a soft gate by design. The session cookie and this one are both
 * client-visible, and Supabase's public signup endpoint is reachable with the
 * anon key from the JS bundle regardless. It stops ordinary visitors, which is
 * what it is for.
 */
export function middleware(request: NextRequest) {
  const { pathname, search } = request.nextUrl;

  if (!pathname.startsWith("/signup")) return NextResponse.next();

  // Someone already signed in has no business being bounced to the gate.
  //
  // The code-verifier exclusion is load-bearing, not defensive. Supabase names
  // the in-flight PKCE cookie `sb-<ref>-auth-token-code-verifier`, which any
  // "starts with sb- and contains auth-token" test matches -- so merely
  // clicking Continue with Google, without ever completing it, would otherwise
  // look like a session and open the gate. Only the real session cookie
  // (`sb-<ref>-auth-token`, possibly chunked .0/.1) should count.
  const signedIn = request.cookies
    .getAll()
    .some(
      (c) =>
        c.name.startsWith("sb-") &&
        c.name.includes("auth-token") &&
        !c.name.includes("code-verifier")
    );

  if (signedIn || request.cookies.get(INVITE_COOKIE)) return NextResponse.next();

  const url = request.nextUrl.clone();
  url.pathname = "/invite";
  // Carry the original destination so the gate can hand them back afterwards.
  url.search = search ? `?next=${encodeURIComponent(pathname + search)}` : "";
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/signup/:path*"],
};
