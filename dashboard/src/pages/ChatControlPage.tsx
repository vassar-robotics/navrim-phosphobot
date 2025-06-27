import { CopilotKit } from "@copilotkit/react-core";
import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { InfoIcon } from "lucide-react";

export default function ChatControlPage() {
  // You can configure the runtime URL based on your backend setup
  // For now, using the cloud runtime as an example
  const runtimeUrl = import.meta.env.VITE_COPILOT_RUNTIME_URL || "http://localhost:5175/chat/copilotkit";

  return (
    <div className="container mx-auto p-4 max-w-6xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">AI Chat Control</h1>
        <p className="text-muted-foreground">
          Control your robot using natural language through an AI-powered chat interface
        </p>
      </div>

      <Alert className="mb-6">
        <InfoIcon className="h-4 w-4" />
        <AlertDescription>
          Chat with the AI assistant to control your robot. You can ask it to perform movements,
          execute tasks, or get information about the robot's current state.
        </AlertDescription>
      </Alert>

      <Card className="h-[calc(100vh-280px)]">
        <CardHeader>
          <CardTitle>Robot Control Assistant</CardTitle>
          <CardDescription>
            Ask the AI to help you control the robot or answer questions about its capabilities
          </CardDescription>
        </CardHeader>
        <CardContent className="h-[calc(100%-120px)] p-0">
          <CopilotKit runtimeUrl={runtimeUrl}>
            <div className="h-full">
              <CopilotChat
                className="h-full"
                labels={{
                  title: "Robot Control Assistant",
                  initial: "Hi! I'm your robot control assistant. How can I help you today?",
                  placeholder: "Ask me to control the robot or about its capabilities..."
                }}
              />
            </div>
          </CopilotKit>
        </CardContent>
      </Card>
    </div>
  );
}