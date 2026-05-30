"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { AuthProvider } from "@/components/auth-provider";
import { Navbar } from "@/components/navbar";
import { Sidebar } from "@/components/sidebar";
import { UploadProgress } from "@/components/upload-progress";
import { AiEditPanel } from "@/components/ai-edit-panel";
import { AiTimelineDock } from "@/components/ai-timeline-dock";
import { useAuthStore } from "@/stores/auth-store";
import { useDriveStore } from "@/stores/drive-store";
import { isSupabaseConfigured } from "@/lib/supabase";

function SetupMessage() {
  return (
    <div className="flex h-screen items-center justify-center px-4">
      <div className="max-w-md space-y-4 text-center">
        <div className="text-5xl">🔧</div>
        <h1 className="text-xl font-bold">Supabase Not Configured</h1>
        <p className="text-sm" style={{ color: "var(--muted)" }}>
          Create a <code className="rounded bg-neutral-100 px-1.5 py-0.5 text-xs dark:bg-neutral-800">frontend/.env.local</code> file with your Supabase credentials:
        </p>
        <pre
          className="rounded-lg p-4 text-left text-xs"
          style={{ background: "var(--sidebar)", border: "1px solid var(--border)" }}
        >
{`NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
NEXT_PUBLIC_API_URL=http://localhost:8000`}
        </pre>
        <p className="text-xs" style={{ color: "var(--muted)" }}>
          Then restart the dev server.
        </p>
      </div>
    </div>
  );
}

function DriveGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const { user, loading } = useAuthStore();

  useEffect(() => {
    if (!loading && !user && isSupabaseConfigured) {
      router.push("/login");
    }
  }, [user, loading, router]);

  if (!isSupabaseConfigured) {
    return <SetupMessage />;
  }

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-current border-t-transparent" style={{ color: "var(--accent)" }} />
      </div>
    );
  }

  if (!user) return null;

  return <>{children}</>;
}

function DriveShell({ children }: { children: React.ReactNode }) {
  const aiPanelOpen = useDriveStore((s) => s.aiPanelOpen);
  const aiScopeFileIds = useDriveStore((s) => s.aiScopeFileIds);
  const closeAiPanel = useDriveStore((s) => s.closeAiPanel);

  return (
    <div className="flex h-screen flex-col">
      <Navbar />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar collapsed={aiPanelOpen} />
        <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
          <div className="flex flex-1 flex-col overflow-y-auto">{children}</div>
          <AiTimelineDock />
        </main>
        {aiPanelOpen && (
          <aside
            className="flex w-[600px] max-w-[55vw] shrink-0 flex-col overflow-hidden border-l"
            style={{ borderColor: "var(--border)", background: "var(--background)" }}
          >
            <AiEditPanel fileIds={aiScopeFileIds} embedded onClose={closeAiPanel} />
          </aside>
        )}
      </div>
    </div>
  );
}

export default function DriveLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <DriveGuard>
        <DriveShell>{children}</DriveShell>
        <UploadProgress />
      </DriveGuard>
    </AuthProvider>
  );
}
