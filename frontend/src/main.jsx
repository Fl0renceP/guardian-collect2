import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'

import './theme.css'
import 'leaflet/dist/leaflet.css'

import Layout from './components/Layout'
import { SessionProvider } from './session'
import HotspotMap from './views/HotspotMap'
import MyClaims from './views/MyClaims'
import ReviewQueue from './views/ReviewQueue'
import SafeRoute from './views/SafeRoute'
import SubmitClaim from './views/SubmitClaim'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <SessionProvider>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<HotspotMap />} />
            <Route path="route" element={<SafeRoute />} />
            <Route path="report" element={<SubmitClaim />} />
            <Route path="my-claims" element={<MyClaims />} />
            <Route path="review" element={<ReviewQueue />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </SessionProvider>
  </React.StrictMode>,
)
