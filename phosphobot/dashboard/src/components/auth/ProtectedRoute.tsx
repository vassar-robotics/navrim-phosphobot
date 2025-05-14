import { useAuth } from "@/context/AuthContext";
import { Navigate } from "react-router-dom";
import { ReactNode } from "react";

export function ProtectedRoute({ children }: { children: ReactNode }) {
    const { session, isLoading } = useAuth();

    if (isLoading) {
        return <div>Loading...</div>;
    }

    if (!session) {
        return <Navigate to="/auth" />;
    }

    return <>{children}</>;
}