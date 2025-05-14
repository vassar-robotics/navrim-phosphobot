import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import CalibrationSequencePage from "@/pages/CalibrationSequencePage";
import JointControlPage from "@/pages/JointControlPage";

export default function CalibrationPage() {
  return (
    <Tabs defaultValue="calibrate">
      <TabsList className="grid w-full grid-cols-2">
        <TabsTrigger value="calibrate">Calibration</TabsTrigger>
        <TabsTrigger value="joint-control">Joints control</TabsTrigger>
      </TabsList>
      <TabsContent value="calibrate">
        <CalibrationSequencePage />
      </TabsContent>
      <TabsContent value="joint-control">
        <JointControlPage />
      </TabsContent>
    </Tabs>
  );
}
