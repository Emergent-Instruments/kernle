import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { AuthProvider } from '@/lib/auth';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Kernle - Stratified Memory for Synthetic Intelligences',
  description: 'Persistent, layered memory infrastructure for AI agents. Give your AI continuity across sessions.',
  icons: {
    icon: '/favicon.png',
    apple: '/icon.png',
  },
  openGraph: {
    title: 'Kernle - Stratified Memory for Synthetic Intelligences',
    description: 'Persistent, layered memory infrastructure for AI agents. Give your AI continuity across sessions.',
    url: 'https://kernle.ai',
    siteName: 'Kernle',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'Kernle - Stratified Memory for Synthetic Intelligences',
      },
    ],
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Kernle - Stratified Memory for Synthetic Intelligences',
    description: 'Persistent, layered memory infrastructure for AI agents.',
    images: ['/og-image.png'],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <AuthProvider>
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
