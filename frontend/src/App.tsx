import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import MainPage from "@/MainPage/MainPage.tsx";

// Main Page Routes 
const MainRoutes = () => {
  return (
    <Routes>
      <Route path="/" element={<MainPage /> } />
    </Routes>
  )
}

function App() {

  return (
    <>
    <BrowserRouter>
        <MainRoutes />

    </BrowserRouter>
    
    </>
    

  )
}

export default App
