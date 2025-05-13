import { Recorder } from "@/components/common/recorder";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useGlobalStore } from "@/lib/hooks";
import { BicepsFlexed, Glasses, Keyboard, Repeat } from "lucide-react";

import KeyboardControlPage from "./KeyboardControlPage";
import LeaderArmPage from "./LeaderArmControlPage";
import ReplayPage from "./SingleArmReplayPage";
import ViewVideo from "./ViewVideoPage";

export default function ControlPage() {
  const showCamera = useGlobalStore((state) => state.showCamera);
  const setShowCamera = useGlobalStore((state) => state.setShowCamera);

  return (
    <div>
      <Tabs defaultValue="keyboard">
        <div className="flex justify-between">
          <TabsList className="flex flex-col md:flex-row gap-4 border-1">
            <TabsTrigger value="keyboard" className="cursor-pointer">
              <Keyboard className="size-4 mr-2" />
              Keyboard control
            </TabsTrigger>
            <TabsTrigger value="leader" className="cursor-pointer">
              <BicepsFlexed className="size-4 mr-2" />
              Leader arm control
            </TabsTrigger>
            <TabsTrigger value="single" className="cursor-pointer">
              <Repeat className="size-4 mr-2" />
              Move with your hands
            </TabsTrigger>
            <TabsTrigger value="VR" className="cursor-pointer">
              <Glasses className="size-4 mr-2" />
              VR control
            </TabsTrigger>
          </TabsList>
          <Recorder showCamera={showCamera} setShowCamera={setShowCamera} />
        </div>
        {showCamera && <ViewVideo />}
        <TabsContent value="keyboard">
          <KeyboardControlPage />
        </TabsContent>
        <TabsContent value="leader">
          <LeaderArmPage />
        </TabsContent>
        <TabsContent value="single">
          <ReplayPage />
        </TabsContent>
        <TabsContent value="VR">
          <div className="flex flex-col gap-4 p-4 bg-background rounded-2xl">
            <div className="text-sm">
              Use the Meta Quest app to control the robot in VR with a Meta
              Quest 2, Meta Quest Pro, Meta Quest 3 or Meta Quest 3s.
            </div>
            <div className="text-sm">
              If you bought a{" "}
              <a
                href="https://robots.phospho.ai/"
                target="_blank"
                className="underline"
              >
                phospho starter pack
              </a>
              , reach out to us on Discord to get access to the app.
            </div>
            <iframe
              width="560"
              height="315"
              src="https://www.youtube.com/embed/AQ-xgCTdj_w?si=tUw1JIWwm75gd5_9"
              title="YouTube video player"
              frame-border="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              referrerPolicy="strict-origin-when-cross-origin"
            ></iframe>
            <div className="text-sm"></div>
            <div className="flex gap-2">
              <Button asChild className="min-w-32" variant="secondary">
                <a href="https://docs.phospho.ai/examples/teleop">Learn more</a>
              </Button>
              <Button asChild className="min-w-32" variant={"secondary"}>
                <a href="https://discord.gg/cbkggY6NSK">Reach out on Discord</a>
              </Button>
              <Button asChild className="min-w-32">
                <a href="https://www.meta.com/en-gb/experiences/phospho-teleoperation/8873978782723478/">
                  Get on the Meta Store
                </a>
              </Button>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
