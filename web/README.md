# Kernle Web Dashboard

A minimal Next.js dashboard for managing Kernle users and API keys.

## Features

- **User Authentication**: Register, login, logout
- **Dashboard**: View account information
- **API Key Management**: Create, list, revoke, and cycle API keys
- **Secure Key Display**: New keys shown once with copy button

## Tech Stack

- Next.js 16 with App Router
- TypeScript
- Tailwind CSS
- shadcn/ui components

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Configure environment:
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your backend URL
   ```

3. Run development server:
   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Kernle backend API URL | `http://localhost:8000` |

## Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page |
| `/login` | User login |
| `/register` | User registration |
| `/dashboard` | Main dashboard (protected) |
| `/dashboard/keys` | API key management (protected) |

## API Integration

The dashboard integrates with the Kernle backend API:

- `POST /auth/register` - Register new user
- `POST /auth/token` - Login (OAuth2 password flow)
- `GET /auth/me` - Get current user
- `GET /keys` - List API keys
- `POST /keys` - Create API key
- `DELETE /keys/{key_id}` - Revoke API key
- `POST /keys/{key_id}/cycle` - Cycle API key

## Production Build

```bash
npm run build
npm start
```
