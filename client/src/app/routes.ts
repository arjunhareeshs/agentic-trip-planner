import { createBrowserRouter } from "react-router";
import Home from "./pages/Home.tsx";
import VibePage from "./pages/VibePage.tsx";
import VibeResultPage from "./pages/VibeResultPage.tsx";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Home,
  },
  {
    path: "/vibe",
    Component: VibePage,
  },
  {
    path: "/result",
    Component: VibeResultPage,
  }
]);