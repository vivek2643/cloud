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

const DEV_USER_ID = "00000000-0000-0000-0000-000000000001";

const DEV_USER = {
  id: DEV_USER_ID,
  email: "dev@local",
  app_metadata: {},
  user_metadata: {},
  aud: "authenticated",
  created_at: new Date(0).toISOString(),
} as unknown as User;

const DEV_SESSION = {
  access_token: "dev-mode-no-auth",
  refresh_token: "dev-mode-no-auth",
  expires_in: 3600,
  expires_at: Math.floor(Date.now() / 1000) + 3600,
  token_type: "bearer",
  user: DEV_USER,
} as unknown as Session;

export const useAuthStore = create<AuthState>((set) => ({
  user: DEV_USER,
  session: DEV_SESSION,
  loading: false,
  setAuth: (user, session) => set({ user, session, loading: false }),
  setLoading: (loading) => set({ loading }),
  clear: () => set({ user: null, session: null, loading: false }),
}));
