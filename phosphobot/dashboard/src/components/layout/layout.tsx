import { ErrorBoundary } from "@/components/common/error";
import { Footer } from "@/components/layout/footer";
import { Sidebar } from "@/components/layout/sidebar";
import { TopBar } from "@/components/layout/topbar";
import { Toaster } from "@/components/ui/sonner";
import { Outlet } from "react-router-dom";

export function Layout() {
  return (
    <div className="flex flex-col min-h-screen bg-muted">
      <TopBar />
      <div className="flex flex-1 mt-[160px] md:mt-[60px] mb-[60px]">
        <Sidebar />
        <main className="container mx-auto py-6 px-4 md:px-6 overflow-y-auto">
          <ErrorBoundary>
            <Outlet />
          </ErrorBoundary>
        </main>
        <Toaster position="top-center" />
      </div>
      <Footer />
    </div>
  );
}
