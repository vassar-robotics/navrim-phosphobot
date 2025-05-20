"use client";

import { AutoComplete, type Option } from "@/components/common/autocomplete";
import { ModelsCard } from "@/components/custom/ModelsDialog";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useGlobalStore } from "@/lib/hooks";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import type { AdminTokenSettings } from "@/types";
import { CheckCircle2, Dumbbell, Lightbulb, List, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import useSWR from "swr";

interface DatasetListResponse {
  pushed_datasets: string[];
  local_datasets: string[];
}

interface TrainingInfoResponse {
  stats: "ok" | "error";
  message?: string;
  training_body: Record<string, unknown>;
}

export default function AITrainingPage() {
  const [selectedDatasetID, setSelectedDataset] = useState<string>("");
  const setSelectedModelType = useGlobalStore(
    (state) => state.setSelectedModelType,
  );
  const [trainingState, setTrainingState] = useState<
    "idle" | "loading" | "success"
  >("idle");
  const selectedModelType = useGlobalStore((state) => state.selectedModelType);
  const { data: adminSettingsTokens } = useSWR<AdminTokenSettings>(
    ["/admin/settings/tokens"],
    ([url]) => fetcher(url, "POST"),
  );
  const { data: datasetsList } = useSWR<DatasetListResponse>(
    ["/dataset/list"],
    ([url]) => fetcher(url, "POST"),
  );
  const { data: datasetInfoResponse, isLoading: isDatasetInfoLoading } =
    useSWR<TrainingInfoResponse>(
      ["/training/info", selectedDatasetID],
      ([url]) =>
        fetcher(url, "POST", {
          model_id: selectedDatasetID,
          model_type: selectedModelType,
        }),
    );

  const [editableJson, setEditableJson] = useState<string>("");

  useEffect(() => {
    // Try to load from localStorage first
    const savedJson = localStorage.getItem("trainingBodyJson");
    if (savedJson) {
      setEditableJson(savedJson);
    }

    // Update from API response when it changes
    if (datasetInfoResponse?.training_body) {
      const jsonString = JSON.stringify(
        datasetInfoResponse.training_body,
        null,
        2,
      );
      setEditableJson(jsonString);
      // Save to localStorage
      localStorage.setItem("trainingBodyJson", jsonString);
    }
  }, [datasetInfoResponse]);

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

      // Parse the edited JSON
      let trainingBody;
      try {
        trainingBody = JSON.parse(editableJson);
      } catch (error) {
        toast.error("Invalid JSON format. Please check your input: " + error, {
          duration: 5000,
        });
        setTrainingState("idle");
        return { success: false, error: "Invalid JSON format" };
      }

      // Send the edited JSON to the training endpoint
      await fetchWithBaseUrl("/train/start", "POST", {
        dataset_name: selectedDatasetID,
        model_name: modelName,
        model_type: selectedModelType,
        training_body: trainingBody,
      });

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
    <div className="container mx-auto py-8">
      <Tabs defaultValue="train">
        <div className="flex justify-between">
          <TabsList className="flex flex-col md:flex-row gap-4 border-1">
            <TabsTrigger value="train">
              <Dumbbell className="size-4 mr-2" />
              Train AI model
            </TabsTrigger>
            <TabsTrigger value="view">
              <List className="size-4 mr-2" />
              View trained models
            </TabsTrigger>
          </TabsList>
        </div>
        <TabsContent value="train">
          <Card className="w-full">
            <CardContent>
              <div className="flex flex-col md:flex-row gap-2 items-end">
                <div className="flex-1/2 flex flex-row md:flex-col gap-2 w-full">
                  <div className="text-xs text-muted-foreground md:w-1/2">
                    Dataset ID on Hugging Face:
                  </div>
                  <AutoComplete
                    key="dataset-autocomplete"
                    options={
                      datasetsList?.pushed_datasets.map((dataset) => ({
                        value: dataset,
                        label: dataset,
                      })) ?? []
                    }
                    value={{
                      value: selectedDatasetID,
                      label: selectedDatasetID,
                    }}
                    onValueChange={(option: Option) => {
                      setSelectedDataset(option.value);
                    }}
                    placeholder="e.g. username/dataset-name"
                    className="w-full"
                    emptyMessage="Make sure this is a public dataset available on Hugging Face."
                  />
                </div>
                <div className="flex-1/4 flex flex-row md:flex-col gap-2 w-full mb-1">
                  <div className="text-xs text-muted-foreground">
                    Type of model to train:
                  </div>
                  <Select
                    defaultValue={selectedModelType}
                    onValueChange={(value) =>
                      setSelectedModelType(value as "gr00t" | "ACT")
                    }
                  >
                    <SelectTrigger className="w-full border rounded-md p-2">
                      <SelectValue placeholder="Select model type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="gr00t">gr00t</SelectItem>
                      <SelectItem value="ACT">ACT</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              {/* Show a text box with the parsed json datasetInfoResposne, the user should be able to edit it */}
              <div className="text-xs text-muted-foreground mt-4">
                Dataset info:
              </div>
              {/* Text should not be overflowing to the right */}
              <div className="text-sm text-muted-foreground mt-2">
                {isDatasetInfoLoading && datasetInfoResponse ? (
                  <div className="flex flex-row items-center">
                    <Loader2 className="size-4 mr-2 animate-spin" />
                    Loading dataset info...
                  </div>
                ) : (
                  <textarea
                    className="bg-gray-100 p-4 rounded-md overflow-x-auto w-full h-64 font-mono text-sm"
                    value={editableJson}
                    onChange={(e) => {
                      setEditableJson(e.target.value);
                      localStorage.setItem("trainingBodyJson", e.target.value);
                    }}
                  />
                )}
              </div>
              <Button
                variant="secondary"
                className="flex width-full mt-4"
                onClick={handleTrainModel}
                disabled={
                  !selectedDatasetID ||
                  trainingState !== "idle" ||
                  isDatasetInfoLoading
                }
              >
                {renderButtonContent()}
              </Button>
              <div className="flex flex-row mt-4">
                <Lightbulb className="size-4 mr-2 text-muted-foreground" />
                Tips
              </div>
              <div className="text-muted-foreground text-sm mt-2">
                If your training fails with a <code>Timeout error</code>, please
                lower the number of steps/epochs.
                <br />
                If your training fails with a{" "}
                <code>Cuda out of memory error</code>, please lower the batch
                size.
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="view">
          <ModelsCard />
        </TabsContent>
      </Tabs>
    </div>
  );
}
