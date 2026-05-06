import { useState } from "react";

interface RegisterProps {
  onLogin: () => void;
  onNavigateToLogin: () => void;
}

export default function Register({ onLogin: _onLogin, onNavigateToLogin }: RegisterProps) {
  const [showPassword, setShowPassword] = useState(false);
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [cafeName, setCafeName] = useState("");
  const [cafeAddress, setCafeAddress] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!fullName || !email || !cafeName || !cafeAddress || !password || !confirmPassword) {
      setError("Please fill in all fields");
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    try {
      const response = await fetch("http://127.0.0.1:8000/api/auth/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          full_name: fullName,
          email: email,
          password: password,
          cafe_name: cafeName,
          cafe_address: cafeAddress,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        setError(data.detail || "Registration failed. Please try again.");
        return;
      }

      onNavigateToLogin();
    } catch (err) {
      console.error(err);
      setError("Cannot connect to the server.");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#f5f3f0] font-sans p-4">
      <div className="w-full max-w-md p-8 bg-white rounded-2xl shadow-xl border border-[#d7cdc1]/50">
        
        <div className="text-center mb-8">
          <h1 className="text-3xl font-extrabold mb-2 text-[#411e03]">Create account</h1>
          <p className="text-[#846851]">Join CoffeeMaps and explore cafes</p>
        </div>

        <form onSubmit={handleRegister}>
          <input
            type="text"
            placeholder="Full name"
            value={fullName}
            onChange={(event) => setFullName(event.target.value)}
            className="w-full p-3 mb-4 rounded-xl bg-white border border-[#d7cdc1] text-[#411e03] focus:outline-none focus:ring-2 focus:ring-[#8a895f] focus:border-transparent transition-all placeholder:text-[#846851]/60"
          />

          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            className="w-full p-3 mb-4 rounded-xl bg-white border border-[#d7cdc1] text-[#411e03] focus:outline-none focus:ring-2 focus:ring-[#8a895f] focus:border-transparent transition-all placeholder:text-[#846851]/60"
          />

          <input
            type="text"
            placeholder="Cafe Name (e.g. Central Perk)"
            value={cafeName}
            onChange={(event) => setCafeName(event.target.value)}
            className="w-full p-3 mb-4 rounded-xl bg-white border border-[#d7cdc1] text-[#411e03] focus:outline-none focus:ring-2 focus:ring-[#8a895f] focus:border-transparent transition-all placeholder:text-[#846851]/60"
          />

          <input
            type="text"
            placeholder="Cafe Address"
            value={cafeAddress}
            onChange={(event) => setCafeAddress(event.target.value)}
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

          <input
            type="password"
            placeholder="Confirm password"
            value={confirmPassword}
            onChange={(event) => setConfirmPassword(event.target.value)}
            className="w-full p-3 mb-6 rounded-xl bg-white border border-[#d7cdc1] text-[#411e03] focus:outline-none focus:ring-2 focus:ring-[#8a895f] focus:border-transparent transition-all placeholder:text-[#846851]/60"
          />

          {error && <p className="text-red-500 text-sm mb-4 text-center">{error}</p>}

          <button
            type="submit"
            className="w-full py-3 rounded-xl bg-[#411e03] hover:bg-[#846851] text-white shadow-md transition-colors font-semibold tracking-wide"
          >
            Create account
          </button>
        </form>

        <p className="text-xs text-[#846851] mt-4 text-center font-medium">
          By signing up, you agree to our Terms & Privacy Policy
        </p>

        <p className="text-center text-sm text-[#846851] mt-6">
          Already have an account?{" "}
          <button
            type="button"
            onClick={onNavigateToLogin}
            className="text-[#8a895f] font-bold hover:text-[#411e03] hover:underline transition-colors"
          >
            Log in
          </button>
        </p>
      </div>
    </div>
  );
}