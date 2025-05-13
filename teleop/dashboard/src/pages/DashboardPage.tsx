import { AIControlDisclaimer } from "@/components/common/ai-control-disclaimer";
import { HuggingFaceKeyInput } from "@/components/common/huggingface-key";
import ModelsDialog from "@/components/custom/ModelsDialog";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useGlobalStore } from "@/lib/hooks";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import { AdminTokenSettings, ServerStatus, TrainingRequest } from "@/types";
import axios from "axios";
import {
  AlertTriangle,
  Bot,
  BrainCircuit,
  Camera,
  CheckCircle2,
  Code,
  Dumbbell,
  FileCog,
  FolderOpen,
  List,
  Loader2,
  LoaderCircle,
  Network,
  Play,
  Settings,
  Sliders,
} from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import useSWR from "swr";

function RobotStatusAlert({
  serverStatus,
  isLoading,
  robotConnected,
}: {
  serverStatus?: ServerStatus;
  isLoading: boolean;
  robotConnected: boolean;
}) {
  if (isLoading) {
    return (
      <Alert>
        <AlertTitle className="flex flex-row gap-1 items-center">
          <LoaderCircle className="animate-spin size-5 mr-1" />
          Status: Loading
        </AlertTitle>
        <AlertDescription>Loading robot status...</AlertDescription>
      </Alert>
    );
  }

  if (!serverStatus) {
    return (
      <Alert>
        <AlertTitle className="flex flex-row gap-1 items-center">
          <span className="size-2 rounded-full bg-red-500" />
          <Bot className="size-5 mr-1" />
          Status: Communication Error
        </AlertTitle>
        <AlertDescription>
          Error fetching robot status. Please check the server connection.
        </AlertDescription>
      </Alert>
    );
  }

  if (robotConnected) {
    return (
      <Alert>
        <AlertTitle className="flex flex-row gap-1 items-center">
          <span className="size-2 rounded-full bg-green-500" />
          <Bot className="size-5 mr-1" />
          Status: Connected
        </AlertTitle>
        <AlertDescription>
          Robot is connected and ready to control.
        </AlertDescription>
      </Alert>
    );
  } else {
    return (
      <Alert>
        <AlertTitle className="flex flex-row gap-1 items-center">
          <span className="size-2 rounded-full bg-red-500" />
          <Bot className="size-5 mr-1" />
          Status: Disconnected
        </AlertTitle>
        <AlertDescription>
          Check the robot is plugged to your computer and powered on. Unplug and
          plug cables again if needed.
        </AlertDescription>
      </Alert>
    );
  }
}

function AIModelsCard() {
  const [showWarning, setShowWarning] = useState(false);
  const [isModelsDialogOpen, setIsModelsDialogOpen] = useState<boolean>(false);
  const [selectedDatasetID, setSelectedDataset] = useState<string | undefined>(
    undefined,
  );
  const [trainingState, setTrainingState] = useState<
    "idle" | "loading" | "success"
  >("idle");
  const selectedModelType = useGlobalStore(
    (state) => state.selectedModelType,
  );
  const setSelectedModelType = useGlobalStore(
    (state) => state.setSelectedModelType,
  );
  const navigate = useNavigate();

  const { data: adminSettingsTokens, isLoading } = useSWR<AdminTokenSettings>(
    ["/admin/settings/tokens"],
    ([url]) => fetcher(url, "POST"),
  );

  const handleControlByAI = () => {
    if (localStorage.getItem("disclaimer_accepted") === "true") {
      navigate(`/inference`);
      return;
    }
    // Otherwise display the warning dialog
    setShowWarning(true);
  };

  const onProceed = () => {
    setShowWarning(false);
    localStorage.setItem("disclaimer_accepted", true.toString());
    navigate(`/inference`);
  };

  const launchModelTraining = async (datasetID: string, modelName: string) => {
    try {
      const trainingRequest: TrainingRequest = {
        dataset_name: datasetID,
        model_name: modelName,
        model_type: selectedModelType,
      };

      await axios.post("/training/start", trainingRequest);
      console.log("Launched training job");
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const statusCode = error.response?.status;

        switch (statusCode) {
          case 429:
            // Don't show toast here, just throw a custom error, we use the error to throw an error toast
            throw new Error(
              "A training job is already in progress. Please wait until it finishes.",
            );
          case 405:
            throw new Error(
              "Training requires a dataset of at least 10 episodes, please record more data.",
            );
          case 404:
            throw new Error("Dataset not found. Please check the ID.");
          case 401:
            navigate("/auth");
            throw new Error(
              "Unauthorized. Please log in to access this feature.",
            );
          case 400:
            console.error("Error training AI model:", error);
            throw new Error(`Error: ${error.message}`);
          case 500:
            console.error("Error training AI model:", error);
            throw new Error(`Internal server error: ${error.message}`);
          default:
            console.error("Error training AI model:", error);
            throw new Error(`Failed to train AI model: ${error.message}`);
        }
      }
      throw new Error("Failed to train AI model");
    }
  };

  const generateHuggingFaceModelName = async (dataset: string) => {
    // Model name followed by 10 random characters
    const randomChars = Math.random().toString(36).substring(2, 12);
    // Remove the name/... and replace with phospho-app/...
    const [, datasetName] = dataset.split("/");

    // Fetch whoami to get the username
    try {
      const result = await fetchWithBaseUrl(
        "/admin/huggingface/whoami",
        "POST",
      );
      // Check the status from the whoami response
      if (result.status === "success" && result.username) {
        // Include username in the model name if status is success
        return `phospho-app/${result.username}-${selectedModelType}-${datasetName}-${randomChars}`;
      } else {
        // Fallback without username if status is not success
        return `phospho-app/${selectedModelType}-${datasetName}-${randomChars}`;
      }
    } catch (error) {
      console.error("Error fetching whoami:", error);
      // Fallback without username in case of error
      return `phospho-app/${selectedModelType}-${datasetName}-${randomChars}`;
    }
  };

  const handleTrainModel = async () => {
    if (!selectedDatasetID) {
      toast.error("Please select a dataset to train the model.", {
        duration: 5000,
      });
      return;
    }

    if (!adminSettingsTokens?.huggingface) {
      toast.error("Please set a valid Hugging Face token in the settings.", {
        duration: 5000,
      });
      return;
    }

    // Set loading state
    setTrainingState("loading");

    try {
      // Generate a random model name
      const modelName = await generateHuggingFaceModelName(selectedDatasetID);
      const modelUrl = `https://huggingface.co/${modelName}`;

      // Send Slack notification and wait for response
      await launchModelTraining(selectedDatasetID, modelName);

      // After successful notification, wait 1 second then show success
      await new Promise((resolve) => setTimeout(resolve, 1000));

      setTrainingState("success");
      toast.success(`Model training started! Check progress at: ${modelUrl}`, {
        duration: 5000,
      });

      return { success: true, modelName };
    } catch (error) {
      console.error("Error starting training job:", error);
      setTrainingState("idle");

      const errorMessage =
        error instanceof Error
          ? error.message
          : "An error occurred while starting the training job. Please try again later.";

      toast.error(errorMessage, {
        duration: 5000,
      });

      return { success: false, error: errorMessage };
    }
  };

  // Render button content based on training state
  const renderButtonContent = () => {
    switch (trainingState) {
      case "loading":
        return (
          <>
            <Loader2 className="size-5 mr-2 animate-spin" />
            Starting...
          </>
        );
      case "success":
        return (
          <>
            <CheckCircle2 className="size-5 mr-2 text-green-500" />
            Training job started
          </>
        );
      default:
        return (
          <>
            <Dumbbell className="size-5 mr-2" />
            Train AI model
          </>
        );
    }
  };

  return (
    <>
      <Card className="flex justify-between md:min-h-[25vh]">
        <CardContent className="flex flex-col md:flex-row py-6">
          <div className="flex-1 md:flex-1/3 mb-4 md:mb-0">
            <div className="flex items-center gap-2 text-xl font-semibold mb-2">
              <BrainCircuit className="text-green-500" />
              AI Training and Control
            </div>
            <div className="text-xs text-muted-foreground">
              Teach your robot new skills. Control your robot with Artificial
              Intelligence.
            </div>
          </div>
          <div className="flex-1 md:flex-2/3">
            <div className="flex flex-col gap-y-4">
              {!isLoading && !adminSettingsTokens?.huggingface && (
                <div className="mb-4">
                  <HuggingFaceKeyInput />
                </div>
              )}
              <div className="flex flex-col md:flex-row gap-2 items-center">
                <div className="flex flex-col md:flex-col gap-2 w-full">
                  <div className="text-xs text-muted-foreground md:w-1/2">
                    HuggingFace dataset ID:
                  </div>
                  <Input
                    key="dataset-id"
                    value={selectedDatasetID}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    placeholder="e.g. username/dataset-name"
                    className="w-full"
                  />
                  <div className="text-xs text-muted-foreground md:w-1/2">
                    Type of model to train:
                  </div>
                  <Select defaultValue={selectedModelType} onValueChange={(value) => setSelectedModelType(value as "gr00t" | "ACT")}>
                    <SelectTrigger className="w-full border rounded-md p-2">
                      <SelectValue placeholder="Select model type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="gr00t">gr00t</SelectItem>
                      <SelectItem value="ACT">ACT</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Button
                  variant="secondary"
                  className="w-full md:w-1/3 cursor-pointer"
                  onClick={handleTrainModel}
                  disabled={!selectedDatasetID || trainingState !== "idle"}
                >
                  {renderButtonContent()}
                </Button>
              </div>

              <div className="flex flex-col md:flex-row gap-2">
                <Button
                  asChild
                  className="flex-1/2"
                  variant="outline"
                  onClick={() => setIsModelsDialogOpen(true)}
                >
                  <a href="#">
                    <List className="size-5" />
                    View Trained Models
                  </a>
                </Button>
                <Tooltip>
                  <TooltipTrigger className="flex-1/2" asChild>
                    <Button
                      onClick={handleControlByAI}
                      className="cursor-pointer"
                      disabled={!adminSettingsTokens?.huggingface}
                    >
                      <BrainCircuit className="size-5" />
                      Go to AI Control
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <div>
                      Once you've trained a model, you can let your AI model
                      control the robot. Make sure your setup is similar to the
                      training environment.
                    </div>
                  </TooltipContent>
                </Tooltip>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
      <Dialog open={showWarning} onOpenChange={setShowWarning}>
        <DialogContent className="sm:max-w-md border-amber-300 border">
          <DialogHeader className="bg-amber-50 dark:bg-amber-950/20 p-4 -m-4 rounded-t-lg">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="size-16 text-red-500 mr-2" />
              <DialogTitle className="text-bold font-bold tracking-tight">
                You are about to surrender control to an artificial intelligence
                system.
              </DialogTitle>
            </div>
          </DialogHeader>

          <AIControlDisclaimer />

          <DialogFooter className="gap-x-2 mt-2">
            <Button
              variant="outline"
              onClick={() => setShowWarning(false)}
              className="border-gray-200 hover:bg-gray-50"
            >
              Cancel
            </Button>
            <Button
              variant="default"
              onClick={onProceed}
              className=" cursor-pointer"
            >
              I Understand the Risks
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      {isModelsDialogOpen && (
        <ModelsDialog
          open={isModelsDialogOpen}
          onOpenChange={setIsModelsDialogOpen}
        />
      )}
    </>
  );
}

export default function DashboardPage() {
  const navigate = useNavigate();
  const { data: serverStatus, isLoading } = useSWR<ServerStatus>(
    ["/status"],
    ([url]) => fetcher(url),
    {
      refreshInterval: 5000,
    },
  );
  const robotConnected =
    serverStatus !== undefined &&
    serverStatus.robots &&
    serverStatus.robots.length > 0;

  return (
    <div className="flex flex-col gap-4">
      {/* Control */}
      <Card className="flex justify-between md:min-h-[25vh]">
        <CardContent className="w-full flex flex-row gap-4">
          <div className="flex-1/3">
            <div className="flex items-center gap-2 text-xl font-semibold mb-2">
              <Play className="text-green-500" />
              Control and Record
            </div>
            <div className="text-xs text-muted-foreground">
              Control the robot with your keyboard, a leader arm, or a VR
              headset. Record and replay movements. Record datasets.
            </div>
          </div>

          <div className="flex-2/3">
            <div className="mb-2 flex flex-col md:flex-row gap-2">
              <div className="flex-1/2">
                <RobotStatusAlert
                  serverStatus={serverStatus}
                  isLoading={isLoading}
                  robotConnected={robotConnected}
                />
              </div>

              <div className="flex-1/2">
                <Button
                  variant="default"
                  className="w-full h-full cursor-pointer"
                  disabled={!robotConnected}
                  onClick={() => {
                    if (!robotConnected) return;
                    navigate("/control");
                  }}
                >
                  <div className="flex items-center gap-2">
                    <Play className="text-green-500" />
                    Control
                  </div>
                </Button>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <Button asChild variant="outline">
                <a href="/browse?path=./lerobot_v2">
                  <FolderOpen className="size-5" />
                  Browse your Datasets
                </a>
              </Button>
              <Button asChild variant="outline">
                <a href="/calibration">
                  <Sliders className="size-5" />
                  Calibration
                </a>
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* AI Models */}
      <AIModelsCard />

      {/* Advanced Settings */}
      <Card className="flex justify-between md:min-h-[25vh]">
        <CardContent className="flex justify-between">
          <div className="flex-1/3">
            <div className="flex items-center gap-2 text-xl font-semibold mb-2">
              <Settings className="text-green-500" />
              Advanced Settings
            </div>
            <div className="text-xs text-muted-foreground">
              Configure the server and the robot settings.
            </div>
          </div>
          <div className="flex-2/3">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              <Button asChild variant="secondary">
                <a href="/admin">
                  <FileCog className="size-5" />
                  Admin Configuration
                </a>
              </Button>
              <Button asChild variant="secondary">
                <a href="/docs">
                  <Code className="size-5" />
                  API Documentation
                </a>
              </Button>

              <Button asChild variant="outline">
                <a href="/viz">
                  <Camera className="size-5" />
                  Camera Overview
                </a>
              </Button>
              <Button asChild variant="outline">
                <a href="/network">
                  <Network className="size-5" />
                  Network Management
                </a>
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
