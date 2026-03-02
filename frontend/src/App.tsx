import { Routes, Route } from 'react-router-dom'
import { Layout } from '@/components/layout/Layout'
import { BacktestPage } from '@/pages/Backtest'
import { EvaluationPage } from '@/pages/Evaluation'
import { VolEvalPage } from '@/pages/VolEval'
import { PriceForecastPage } from '@/pages/PriceForecast'
import { CoMovPage } from '@/pages/CoMov'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<BacktestPage />} />
        <Route path="/evaluation" element={<EvaluationPage />} />
        <Route path="/vol-eval" element={<VolEvalPage />} />
        <Route path="/price-forecast" element={<PriceForecastPage />} />
        <Route path="/co-mov" element={<CoMovPage />} />
      </Route>
    </Routes>
  )
}
