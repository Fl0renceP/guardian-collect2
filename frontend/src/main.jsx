import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'

import './theme.css'
import 'leaflet/dist/leaflet.css'

import Layout from './components/Layout'
import { SessionProvider } from './session'
import AlertsFeed from './views/AlertsFeed'
import HotspotMap from './views/HotspotMap'
import MemberProfile from './views/MemberProfile'
import MyClaims from './views/MyClaims'
import PatrolPlan from './views/PatrolPlan'
import ReviewQueue from './views/ReviewQueue'
import MemberSafetyScore from './views/MemberSafetyScore'
import SafeRoute from './views/SafeRoute'
import SubmitClaim from './views/SubmitClaim'
import LiveScanDemo from './components/LiveScanDemo' // Import your live scan demo component

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
            <Route path="profile" element={<MemberProfile />} />
            <Route path="review" element={<ReviewQueue />} />
            <Route path="safety-score" element={<MemberSafetyScore />} />
            <Route path="alerts" element={<AlertsFeed />} />
            <Route path="patrol" element={<PatrolPlan />} />
            <Route path="live-scan" element={<LiveScanDemo />} /> {/* Added route */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </SessionProvider>
  </React.StrictMode>,
)