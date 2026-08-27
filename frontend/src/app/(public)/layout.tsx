// frontend_project_ux.plan.md Stage 2.5: a bare shell for anonymous, public
// pages -- deliberately OUTSIDE the (drive) route group, so none of
// DriveGuard's auth redirect or DriveShell's navbar/sidebar/panels apply.
// The root layout (app/layout.tsx) still provides <html>/<body> and
// globals.css; this adds nothing beyond a full-height container.
export default function PublicLayout({ children }: { children: React.ReactNode }) {
  return <div className="min-h-screen">{children}</div>;
}
