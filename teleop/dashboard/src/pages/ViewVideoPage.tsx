import { CameraStreamCard } from "@/components/common/camera-stream-card";
import { useCameraControls } from "@/lib/hooks";
import { fetcher } from "@/lib/utils";
import type { AdminSettings, ServerStatus } from "@/types";
import { Video } from "lucide-react";
import useSWR from "swr";

export default function ViewVideo({ labelText }: { labelText?: string }) {
  if (!labelText) labelText = "Camera Stream";

  const { data: serverStatus } = useSWR<ServerStatus>(["/status"], fetcher, {
    refreshInterval: 5000,
  });

  const { data: adminSettings, mutate: mutateSettings } = useSWR<AdminSettings>(
    "/admin/settings",
    fetcher,
    {
      revalidateOnFocus: false,
      revalidateOnReconnect: false,
    },
  );

  const { updateCameraRecording, isCameraEnabled } = useCameraControls(
    adminSettings,
    mutateSettings,
  );

  return (
    <>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
        {serverStatus?.cameras.video_cameras_ids.map((cameraId) => {
          return (
            <CameraStreamCard
              key={cameraId}
              id={cameraId}
              title={`Camera ${cameraId}`}
              streamPath={`/video/${cameraId}`}
              alt={`Video Stream ${cameraId}`}
              icon={<Video className="h-4 w-4" />}
              isRecording={isCameraEnabled(cameraId)}
              onRecordingToggle={updateCameraRecording}
              showRecordingControls={true}
              labelText={labelText}
            />
          );
        })}
      </div>
    </>
  );
}
