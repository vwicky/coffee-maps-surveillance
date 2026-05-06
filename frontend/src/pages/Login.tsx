import { useState } from "react";

interface LoginProps {
  onLogin: () => void;
  onNavigateToRegister: () => void;
}

const API_ROOT = "http://localhost:8000";

export default function Login({ onLogin, onNavigateToRegister }: LoginProps) {
  const [showPassword, setShowPassword] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleLogin = async () => {
    setError("");
    setIsSubmitting(true);

    try {
      const response = await fetch(`${API_ROOT}/api/auth/login`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email,
          password,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Не вдалося увійти.");
      }

      localStorage.setItem("cafe_token", data.access_token);
      onLogin();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не вдалося увійти.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#f5f3f0] font-sans p-4">
      <div className="w-full max-w-md p-8 bg-white rounded-2xl shadow-xl border border-[#d7cdc1]/50">
        
        <div className="text-center mb-8">
          <h1 className="text-3xl font-extrabold mb-2 text-[#411e03]">Welcome back</h1>
          <p className="text-[#846851]">Log in to your CoffeeMaps account</p>
        </div>

        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          className="w-full p-3 mb-4 rounded-xl bg-white border border-[#d7cdc1] text-[#411e03] focus:outline-none focus:ring-2 focus:ring-[#8a895f] focus:border-transparent transition-all placeholder:text-[#846851]/60"
        />

        <div className="relative mb-4">
          <input
            type={showPassword ? "text" : "password"}
            placeholder="Password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="w-full p-3 rounded-xl bg-white border border-[#d7cdc1] text-[#411e03] focus:outline-none focus:ring-2 focus:ring-[#8a895f] focus:border-transparent transition-all placeholder:text-[#846851]/60"
          />
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className="absolute right-4 top-3 text-sm font-medium text-[#846851] hover:text-[#411e03] transition-colors"
          >
            {showPassword ? "Hide" : "Show"}
          </button>
        </div>

        <div className="text-right mb-6">
          <button type="button" className="text-sm text-[#8a895f] hover:text-[#411e03] hover:underline font-medium transition-colors">
            Forgot password?
          </button>
        </div>

        {error && (
          <p className="mb-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
            {error}
          </p>
        )}

        <button 
          type="button"
          onClick={() => void handleLogin()}
          disabled={isSubmitting}
          className="w-full py-3 rounded-xl bg-[#411e03] hover:bg-[#846851] text-white shadow-md transition-colors font-semibold tracking-wide"
        >
          {isSubmitting ? "Signing in..." : "Log in"}
        </button>

        <div className="flex items-center my-6">
          <div className="flex-1 h-px bg-[#d7cdc1]"></div>
          <span className="px-3 text-[#846851] text-sm font-medium">or</span>
          <div className="flex-1 h-px bg-[#d7cdc1]"></div>
        </div>

        <button type="button" className="w-full py-3 mb-6 rounded-xl bg-white text-[#411e03] border border-[#d7cdc1] font-medium hover:bg-[#f5f3f0] transition-all flex items-center justify-center gap-2 shadow-sm">
          <svg className="w-5 h-5" viewBox="0 0 24 24">
            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
          </svg>
          Continue with Google
        </button>

        <p className="text-center text-sm text-[#846851]">
          Don’t have an account?{" "}
          <button
            type="button"
            onClick={onNavigateToRegister}
            className="text-[#8a895f] font-bold hover:text-[#411e03] hover:underline transition-colors"
          >
            Sign up
          </button>
        </p>
      </div>
    </div>
  );
}