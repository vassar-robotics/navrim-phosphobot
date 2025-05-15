import { Session } from "@/types";
import { createContext, useContext, useState, useEffect, ReactNode } from "react";


interface AuthContextType {
    session: Session | null;
    isLoading: boolean;
    login: (email: string, password: string, session?: Session) => Promise<void>;
    signup: (email: string, password: string) => Promise<void>;
    logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
    const [session, setSession] = useState<Session | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(true);

    useEffect(() => {
        const storedSession = localStorage.getItem("session");
        if (storedSession) {
            setSession(JSON.parse(storedSession));
        }
        setIsLoading(false);
    }, []);

    const login = async (
        email: string,
        password: string,
        directSession?: Session
    ): Promise<void> => {
        if (directSession) {
            // Direct session from email confirmation
            setSession(directSession);
            localStorage.setItem("session", JSON.stringify(directSession));
            return;
        }

        const response = await fetch("/auth/signin", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password }),
        });
        if (!response.ok) {
            throw new Error("Login failed");
        }
        const data: { message: string; session: Session } = await response.json();
        localStorage.setItem("session", JSON.stringify(data.session));
        setSession(data.session);
    };

    const signup = async (email: string, password: string): Promise<void> => {
        const response = await fetch("/auth/signup", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password }),
        });
        if (!response.ok) {
            throw new Error("Signup failed");
        }
        const data: { message: string; session?: Session } = await response.json();
        if (data.session) {
            localStorage.setItem("session", JSON.stringify(data.session));
            setSession(data.session);
        }
    };

    const logout = async (): Promise<void> => {
        await fetch("/auth/logout", { method: "POST" });
        localStorage.removeItem("session");
        setSession(null);
    };

    const validateSession = async () => {
        try {
            const response = await fetch("/auth/check_auth", {
                headers: {
                    "Authorization": `Bearer ${session?.access_token}`,
                },
            });
            if (!response.ok) {
                throw new Error("Session invalid");
            }
            const data = await response.json();
            if (!data.authenticated) {
                logout();
            }
        } catch (e) {
            console.error("Session validation failed:", e);
            logout();
        }
    };

    useEffect(() => {
        if (session) {
            validateSession();
        }
    }, [isLoading, session]);

    return (
        <AuthContext.Provider value={{ session, isLoading, login, signup, logout }}>
            {children}
        </AuthContext.Provider>
    );
}

export function useAuth(): AuthContextType {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error("useAuth must be used within an AuthProvider");
    }
    return context;
}