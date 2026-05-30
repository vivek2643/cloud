"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { AiEditPanel } from "@/components/ai-edit-panel";

function EditChatInner() {
  const searchParams = useSearchParams();
  const ids = (searchParams.get("file_ids") || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  return <AiEditPanel fileIds={ids} embedded={false} />;
}

export default function EditChatPage() {
  return (
    <Suspense fallback={null}>
      <EditChatInner />
    </Suspense>
  );
}
