import { create } from "zustand";
import type { User, Session } from "@supabase/supabase-js";

interface AuthState {
  user: User | null;
  session: Session | null;
  loading: boolean;
  setAuth: (user: User | null, session: Session | null) => void;
  setLoading: (loading: boolean) => void;
  clear: () => void;
}

// Starts empty and loading, NOT with a stand-in user.
//
// This used to seed a fake session carrying access_token "dev-mode-no-auth",
// from the era when the backend honoured DEV_USER_ID and ignored the token. It
// was harmless then. Once real JWT verification shipped it became a live bug:
// every component reads session.access_token, so anything firing before
// AuthProvider's getSession() resolved sent that string as a bearer token and
// the API answered 401 "Invalid token: Not enough segments" -- which is what
// broke loading the drive, and with it exports.
//
// loading: true is the other half. Callers gate on it (see DriveGuard) to wait
// for the real session instead of acting on a placeholder.
export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  session: null,
  loading: true,
  setAuth: (user, session) => set({ user, session, loading: false }),
  setLoading: (loading) => set({ loading }),
  clear: () => set({ user: null, session: null, loading: false }),
}));
