import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'JINALYS AI — Хаттама көмекшісі', description: 'Жиналыстан нақты тапсырмаларға' };
export default function RootLayout({ children }: Readonly<{children: React.ReactNode}>) { return <html lang="kk"><body>{children}</body></html>; }
