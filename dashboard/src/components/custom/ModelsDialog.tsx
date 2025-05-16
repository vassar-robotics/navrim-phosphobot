"use client";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { fetcher } from "@/lib/utils";
import type { SupabaseTrainingModel, TrainingConfig } from "@/types";
import { Check, Copy, ExternalLink, Loader2, X } from "lucide-react";
import { useState } from "react";
import type React from "react";
import useSWR from "swr";

interface ModelsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CopyButton({ text, hint }: { text: string; hint: string }) {
  const [isCopied, setIsCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 1500); // Hide the tooltip after 1.5 seconds
  };

  return (
    <TooltipProvider>
      <Tooltip open={isCopied}>
        <TooltipTrigger asChild>
          <Button
            onClick={handleCopy}
            title={hint}
            aria-label={hint}
            variant="ghost"
            size="icon"
          >
            <Copy className="h-4 w-4" />
            <span className="sr-only">Copy</span>
          </Button>
        </TooltipTrigger>
        <TooltipContent>Copied!</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

const ValueWithTooltip = ({ value }: { value: string }) => {
  // Don't use tooltip if value is short
  if (!value || value.length < 25) {
    return <span>{value}</span>;
  }

  // Show a preview of the first 20 characters after the last "/"
  const preview = value.split("/").pop()?.slice(0, 20) + "..." || value;

  return (
    <TooltipProvider>
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <span className="border-b border-dotted border-muted-foreground">
            {preview}
          </span>
        </TooltipTrigger>
        <TooltipContent className="max-w-md break-words">
          {value}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

// ModelRow component
const ModelRow: React.FC<{ model: SupabaseTrainingModel }> = ({ model }) => {
  // Set uppercase status for display
  const status = model.status.charAt(0).toUpperCase() + model.status.slice(1);

  // Define model url
  const url = "https://huggingface.co/" + model.model_name;

  return (
    <>
      <TableRow>
        <TableCell>
          {status === "Succeeded" && (
            <Check className="h-4 w-4 inline mr-1 text-green-500" />
          )}
          {status === "Running" && (
            <Loader2 className="h-4 w-4 animate-spin inline mr-1" />
          )}
          {(status === "Failed" || status === "Canceled") && (
            <X className="h-4 w-4 inline mr-1 text-red-500" />
          )}
          {status}
        </TableCell>
        <TableCell>
          <div className="flex items-center flex-row justify-between">
            {ValueWithTooltip({ value: model.model_name })}

            <div className="flex items-center">
              <CopyButton text={model.model_name} hint={"Copy model name"} />
              {/* Button to open model page */}
              <Button
                onClick={() => window.open(url, "_blank")}
                title="Go to model"
                aria-label="Go to model"
                className="text-blue-500 hover:bg-blue-50 cursor-pointer"
                variant="ghost"
                size="icon"
              >
                <ExternalLink className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </TableCell>
        <TableCell>{model.model_type}</TableCell>
        <TableCell>
          <div className="flex flex-col">
            {model.training_params &&
              Object.entries(model.training_params).map(
                ([key, value]) =>
                  value !== null && (
                    <div key={key}>
                      <strong>{key}:</strong> {String(value)}
                    </div>
                  ),
              )}
          </div>
        </TableCell>
        <TableCell>{ValueWithTooltip({ value: model.dataset_name })}</TableCell>
        <TableCell>
          {model.used_wandb ? (
            <Check className="h-4 w-4 inline mr-1 text-green-500" />
          ) : (
            <X className="h-4 w-4 inline mr-1 text-red-500" />
          )}
          {model.used_wandb ? "Yes" : "No"}
        </TableCell>
        <TableCell>{new Date(model.requested_at).toLocaleString()}</TableCell>
      </TableRow>
    </>
  );
};

export const ModelsDialog: React.FC<ModelsDialogProps> = ({
  open,
  onOpenChange,
}) => {
  const {
    data: modelsData,
    isLoading,
    error,
  } = useSWR<TrainingConfig>(
    ["/training/models/read"],
    ([endpoint]) => fetcher(endpoint, "POST"),
    {
      refreshInterval: 5000,
    },
  );

  const models = modelsData?.models || [];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[1200px] w-full">
        <DialogHeader>
          <DialogTitle>Trained Models</DialogTitle>
          <DialogDescription>
            You can only run one training job at a time.
            <br />
            If you get a "Failed" status, please check the error log on the
            Hugging Face model page.
            <br />
            For more advanced options, such as changing the number of epochs,
            steps, etc..., please use the <code>/training/start</code> api
            endpoint.
          </DialogDescription>
        </DialogHeader>

        {isLoading ? (
          <div className="flex justify-center py-4">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center py-4">
            <p className="text-red-500">{error}</p>
          </div>
        ) : models.length === 0 ? (
          <div className="text-center py-4">No models found.</div>
        ) : (
          <div className="max-h-[400px] overflow-auto border rounded-md">
            <Table>
              <TableHeader className="sticky top-0 bg-white z-10">
                <TableRow>
                  <TableHead>Status</TableHead>
                  <TableHead>Model Name</TableHead>
                  <TableHead>Model Type</TableHead>
                  <TableHead>Training Parameters</TableHead>
                  <TableHead>Dataset Name</TableHead>
                  <TableHead>Wandb</TableHead>
                  <TableHead>Created at</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {models.map((model, index) => (
                  <ModelRow key={index} model={model} />
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export const ModelsCard: React.FC = () => {
  const {
    data: modelsData,
    isLoading,
    error,
  } = useSWR<TrainingConfig>(
    ["/training/models/read"],
    ([endpoint]) => fetcher(endpoint, "POST"),
    {
      refreshInterval: 5000,
    },
  );

  const models = modelsData?.models || [];

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Trained Models</CardTitle>
        <CardDescription>
          You can only run one training job at a time.
          <br />
          If you get a "Failed" status, please check the error log on the
          Hugging Face model page.
          <br />
          For more advanced options, such as changing the number of epochs,
          steps, etc..., please use the <code>/training/start</code> api
          endpoint.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex justify-center py-4">
            <Loader2 className="h-6 w-6 animate-spin" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center py-4">
            <p className="text-red-500">{error.toString()}</p>
          </div>
        ) : models.length === 0 ? (
          <div className="text-center py-4">No models found.</div>
        ) : (
          <div className="max-h-[400px] overflow-auto border rounded-md">
            <Table>
              <TableHeader className="sticky top-0 bg-white z-10">
                <TableRow>
                  <TableHead>Status</TableHead>
                  <TableHead>Model Name</TableHead>
                  <TableHead>Model Type</TableHead>
                  <TableHead>Training Parameters</TableHead>
                  <TableHead>Dataset Name</TableHead>
                  <TableHead>Wandb</TableHead>
                  <TableHead>Created at</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {models.map((model, index) => (
                  <ModelRow key={index} model={model} />
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
