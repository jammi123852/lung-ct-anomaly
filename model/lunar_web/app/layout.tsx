import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'LUNAR - Lung Anomaly Recognition',
  description: 'AI-powered CT scan analysis for lung anomaly detection',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="bg-[#121212]">
      <body className="font-sans antialiased overflow-hidden">
        {children}
      </body>
    </html>
  )
}
