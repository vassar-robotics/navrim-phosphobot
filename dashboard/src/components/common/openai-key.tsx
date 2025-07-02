import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  AlertCircle,
  CheckCircle2,
  HelpCircle,
  LoaderCircle,
  Save,
} from "lucide-react";
import type React from "react";
import { useState } from "react";

export function OpenAIKeyInput() {
  const [apiKey, setApiKey] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Handle OpenAI form submission
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    // Reset states
    setError(null);
    setSuccess(false);

    // Validate API key
    if (!apiKey.trim()) {
      setError("OpenAI API Key is required");
      return;
    }

    if (!apiKey.startsWith("sk-")) {
      setError("API Key should start with 'sk-'");
      return;
    }

    setIsLoading(true);

    try {
      const response = await fetch("/admin/openai", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ token: apiKey }),
      });

      const result = await response.json();

      if (response.ok && result.status == "success") {
        setSuccess(true);
      } else {
        setError(result.message || "Failed to save API key");
      }
    } catch (error) {
      console.error("Error saving API key:", error);
      setError("An unexpected error occurred. Please try again.");
    } finally {
      setIsLoading(false);

      // Auto-hide success message after 5 seconds
      if (success) {
        setTimeout(() => {
          setSuccess(false);
        }, 5000);
      }
    }
  };

  return (
    <div className="space-y-2">
      <form onSubmit={handleSubmit} className="space-y-2">
        <div className="space-y-2">
          <Label htmlFor="apiKey">
            OpenAI API Key{" "}
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <HelpCircle className="h-4 w-4 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs">
                    Your API key is securely stored. It will be used to enable
                    AI-powered features and chat functionality.
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </Label>
          <div className="text-sm text-muted-foreground">
            <p>
              Go to{" "}
              <a
                href="https://platform.openai.com/api-keys"
                target="_blank"
                rel="noopener noreferrer"
                className="underline text-primary hover:text-primary/80"
              >
                OpenAI API keys page
              </a>{" "}
              and create a new API key. Make sure you have{" "}
              <span className="font-semibold">
                sufficient credits or an active subscription
              </span>{" "}
              for using OpenAI services.
            </p>
          </div>
          <div className="flex gap-x-2">
            <Input
              id="apiKey"
              type="password"
              placeholder="sk-••••••••••••••••••••••••••••••"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className={
                error ? "border-red-500 focus-visible:ring-red-500" : ""
              }
              disabled={isLoading}
              autoComplete="off"
            />
            <Button
              type="submit"
              disabled={isLoading}
              className="cursor-pointer"
            >
              {isLoading ? (
                <span className="flex items-center">
                  <LoaderCircle className="animate-spin size-5" />
                  Saving...
                </span>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  Save token
                </>
              )}
            </Button>
          </div>
        </div>
      </form>

      {error && (
        <Alert variant="destructive" className="mt-2">
          <AlertCircle className="h-4 w-4" />
          <div className="ml-2">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </div>
        </Alert>
      )}

      {success && (
        <Alert className="mt-2 bg-green-50 text-green-800 border-green-200">
          <CheckCircle2 className="h-4 w-4 text-green-600" />
          <div className="ml-2">
            <AlertTitle>Success</AlertTitle>
            <AlertDescription>
              OpenAI API key has been saved successfully.
            </AlertDescription>
          </div>
        </Alert>
      )}
    </div>
  );
}
