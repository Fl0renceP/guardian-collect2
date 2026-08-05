import { api } from './api'

// PushManager needs the VAPID public key as a raw Uint8Array, not base64url text.
function urlBase64ToUint8Array(base64) {
  const padding = '='.repeat((4 - (base64.length % 4)) % 4)
  const raw = atob((base64 + padding).replace(/-/g, '+').replace(/_/g, '/'))
  return Uint8Array.from([...raw].map((c) => c.charCodeAt(0)))
}

export function pushSupported() {
  return 'serviceWorker' in navigator && 'PushManager' in window
}

export async function enablePushNotifications(userId) {
  if (!pushSupported() || !userId) return false

  const permission = await Notification.requestPermission()
  if (permission !== 'granted') return false

  const registration = await navigator.serviceWorker.register('/sw.js')
  const { public_key: publicKey } = await api.pushPublicKey()
  if (!publicKey) throw new Error('Push is not configured on the server yet.')

  let subscription = await registration.pushManager.getSubscription()
  if (!subscription) {
    subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(publicKey),
    })
  }
  await api.pushSubscribe(userId, subscription.toJSON())
  return true
}

export async function disablePushNotifications(userId) {
  if (!pushSupported()) return
  const registration = await navigator.serviceWorker.getRegistration()
  const subscription = await registration?.pushManager.getSubscription()
  if (!subscription) return
  await api.pushUnsubscribe(userId, subscription.endpoint)
  await subscription.unsubscribe()
}
