import { CameraStreamCard } from "@/components/common/camera-stream-card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useGlobalStore } from "@/lib/hooks";
import { fetcher } from "@/lib/utils";
import type { ServerStatus } from "@/types";
import { TriangleAlert } from "lucide-react";
import { useEffect } from "react";
import useSWR from "swr";

export interface CameraKeyMapperProps {
  modelKeys?: string[];
  onMappingChange?: (mapping: Record<string, number> | null) => void;
}

export default function CameraKeyMapper({
  modelKeys,
  onMappingChange,
}: CameraKeyMapperProps) {
  const { data: serverStatus } = useSWR<ServerStatus>(["/status"], fetcher, {
    refreshInterval: 5000,
  });

  const cameraIds = serverStatus?.cameras.video_cameras_ids || [];
  const mapping = useGlobalStore((state) => state.cameraKeysMapping);
  const setMapping = useGlobalStore((state) => state.setCameraKeysMapping);

  useEffect(() => {
    if (modelKeys === undefined) {
      setMapping(null);
      return;
    }

    if (cameraIds.length > 0) {
      const initial: Record<string, number> = {};
      modelKeys.forEach((key, idx) => {
        initial[key] =
          idx < cameraIds.length
            ? cameraIds[idx]
            : cameraIds[cameraIds.length - 1];
      });
      setMapping(initial);
    } else {
      setMapping(null);
    }
  }, [cameraIds, modelKeys]);

  useEffect(() => {
    onMappingChange?.(mapping);
  }, [mapping, onMappingChange]);

  const handleChange = (modelKey: string, camId: number) => {
    const newValue = mapping
      ? { ...mapping, [modelKey]: camId }
      : { [modelKey]: camId };
    setMapping(newValue);
  };

  if (!serverStatus) {
    return <p>Loading camera list...</p>;
  }

  return (
    <>
      <h2 className="text-lg font-semibold mb-4">Camera Mapping</h2>
      <p className="text-sm text-muted-foreground mb-4">
        Select the camera to be sent to each model key. The camera angles should
        match the dataset used for training the model.
      </p>
      {modelKeys === undefined ||
        (modelKeys?.length === 0 && (
          <p className="mb-4 flex items-center flex-row">
            <TriangleAlert className="mr-2 h-4 w-4" />
            <p className="text-sm text-muted-foreground">
              No video keys available. Please enter a model name first and make
              sure it's valid.
            </p>
          </p>
        ))}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
        {modelKeys !== undefined &&
          mapping !== null &&
          modelKeys.map((key) => {
            const camId = mapping[key];
            return (
              <div key={key} className="flex flex-col gap-2">
                <div>
                  <div className="text-lg font-medium">
                    <code>{key}</code>
                  </div>
                  <Select
                    value={camId?.toString() ?? ""}
                    onValueChange={(value) =>
                      handleChange(key, parseInt(value, 10))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select camera" />
                    </SelectTrigger>
                    <SelectContent>
                      {cameraIds.map((id) => (
                        <SelectItem key={id} value={id.toString()}>
                          Camera {id}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                {camId !== undefined && (
                  <CameraStreamCard
                    id={camId}
                    title={`Camera ${camId}`}
                    streamPath={`/video/${camId}`}
                    alt={`Video Stream ${camId}`}
                    showRecordingControls={false}
                  />
                )}
                {camId === undefined && (
                  <div className="flex items-center justify-center h-32 bg-muted rounded-md">
                    <p className="text-sm text-muted-foreground">
                      <TriangleAlert className="mr-2 h-4 w-4 text-red-500" />
                      No camera selected!
                    </p>
                  </div>
                )}
              </div>
            );
          })}
      </div>
    </>
  );
}
