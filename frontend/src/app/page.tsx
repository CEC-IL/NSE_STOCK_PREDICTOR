
"use client";

import { useMemo, useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Bell,
  ChartNoAxesCombined,
  ChevronDown,
  CircleHelp,
  LayoutDashboard,
  Menu,
  Search,
  Settings,
  Star,
  TrendingUp,
  Wallet,
  X,
} from "lucide-react";
import { stocks } from "@/data/stocks";

function formatKES(value: number) {
  return new Intl.NumberFormat("en-KE", {
    style: "currency",
    currency: "KES",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

function formatVolume(value: number) {
  return new Intl.NumberFormat("en-KE", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}
{/*mini chart*/}
function MiniChart({
  values,
  positive,
}: {
  values: number[];
  positive: boolean;
}) {
  const width = 120;
  const height = 38;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const points = values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * width;
      const y = height - ((value - min) / range) * (height - 5) - 2;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="mini-chart"
      role="img"
      aria-label="Illustrative recent price trend"
    >
      <polyline
        points={points}
        fill="none"
        stroke={positive ? "#16a34a" : "#dc2626"}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function Home() {
  const [search, setSearch] = useState("");
  const [sector, setSector] = useState("All sectors");
  const [sortBy, setSortBy] = useState("name");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [watchlist, setWatchlist] = useState<string[]>([]);

  const sectors = [
    "All sectors",
    ...Array.from(new Set(stocks.map((stock) => stock.sector))),
  ];
{/*filter stock */}
  const filteredStocks = useMemo(() => {
    const query = search.trim().toLowerCase();

    return stocks
      .filter((stock) => {
        const matchesSearch =
          stock.name.toLowerCase().includes(query) ||
          stock.symbol.toLowerCase().includes(query);

        const matchesSector =
          sector === "All sectors" || stock.sector === sector;

        return matchesSearch && matchesSector;
      })
      .sort((a, b) => {
        if (sortBy === "price") return b.price - a.price;
        if (sortBy === "change") return b.change - a.change;
        return a.name.localeCompare(b.name);
      });
  }, [search, sector, sortBy]);

{/*gainers and losers*/}
  const gainers = [...stocks]
    .filter((stock) => stock.change > 0)
    .sort((a, b) => b.change - a.change);

  const losers = [...stocks]
    .filter((stock) => stock.change < 0)
    .sort((a, b) => a.change - b.change);

    
{/* Watchlist */}
  function toggleWatchlist(symbol: string) {
    setWatchlist((current) =>
      current.includes(symbol)
        ? current.filter((item) => item !== symbol)
        : [...current, symbol]
    );
  }

  return (
    <div className="app-shell">
      {mobileMenuOpen && (
        <button
          className="mobile-backdrop"
          aria-label="Close navigation"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      <aside className={`sidebar ${mobileMenuOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-icon">
            <ChartNoAxesCombined size={23} />
          </div>
          <div>
            <strong>JENGA</strong>
            <span>MARKET INTELLIGENCE</span>
          </div>
          <button
            className="icon-button sidebar-close"
            onClick={() => setMobileMenuOpen(false)}
            aria-label="Close menu"
          >
            <X size={19} />
          </button>
        </div>

        <div className="market-label">WORKSPACE</div>

        <nav className="sidebar-nav">
          <a className="nav-link active" href="#overview">
            <LayoutDashboard size={18} />
            <span>Market overview</span>
          </a>
          <a className="nav-link" href="#stocks">
            <TrendingUp size={18} />
            <span>Stock screener</span>
          </a>
          <a className="nav-link" href="#watchlist">
            <Star size={18} />
            <span>My watchlist</span>
            <span className="nav-count">{watchlist.length}</span>
          </a>
          <a className="nav-link" href="#market-movers">
            <Activity size={18} />
            <span>Market movers</span>
          </a>
        </nav>

        <div className="market-label">YOUR ACCOUNT</div>
        <nav className="sidebar-nav">
          <a className="nav-link" href="#portfolio">
            <Wallet size={18} />
            <span>Portfolio</span>
          </a>
          <a className="nav-link" href="#alerts">
            <Bell size={18} />
            <span>Price alerts</span>
          </a>
          <a className="nav-link" href="#settings">
            <Settings size={18} />
            <span>Settings</span>
          </a>
        </nav>

        <div className="sidebar-bottom">
          <div className="help-icon">
            <CircleHelp size={19} />
          </div>
          <div>
            <strong>New to investing?</strong>
            <p>Learn how to read the market.</p>
          </div>
        </div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <button
            className="icon-button mobile-menu-button"
            onClick={() => setMobileMenuOpen(true)}
            aria-label="Open navigation"
          >
            <Menu size={21} />
          </button>

          <div className="breadcrumb">
            <span>Markets</span>
            <span>/</span>
            <strong>Nairobi Securities Exchange</strong>
          </div>

          <div className="topbar-right">
            <span className="demo-pill">DEMO DATA</span>
            <button className="icon-button" aria-label="Notifications">
              <Bell size={19} />
            </button>
            <div className="avatar">JM</div>
          </div>
        </header>

        <div className="page-content">
          <section className="welcome-row">
            <div>
              <div className="eyebrow">
                <span className="status-dot" />
                NSE MARKET EXPLORER
              </div>
              <h1>Market overview</h1>
              <p className="subtitle">
                Explore Kenyan equities and understand market movements.
              </p>
            </div>

            <div className="date-label">
              <Activity size={16} />
              Historical demo
              <ChevronDown size={15} />
            </div>
          </section>

          <div className="notice">
            <Activity size={18} />
            <p>
              <strong>Demonstration mode.</strong> All prices, changes, volumes
              and charts on this page are illustrative, not actual NSE market
              data or investment advice.
            </p>
          </div>

          <section className="metrics-grid" aria-label="Market summary">
            <article className="metric-card">
              <div className="metric-top">
                <span>Companies displayed</span>
                <span className="metric-icon blue">
                  <LayoutDashboard size={18} />
                </span>
              </div>
              <div className="metric-value">{stocks.length}</div>
              <div className="metric-caption">Illustrative company list</div>
            </article>

            <article className="metric-card">
              <div className="metric-top">
                <span>Positive movers</span>
                <span className="metric-icon green">
                  <ArrowUpRight size={18} />
                </span>
              </div>
              <div className="metric-value">{gainers.length}</div>
              <div className="metric-caption positive-text">
                <TrendingUp size={14} />
                Demo change above 0%
              </div>
            </article>

            <article className="metric-card">
              <div className="metric-top">
                <span>Negative movers</span>
                <span className="metric-icon red">
                  <ArrowDownRight size={18} />
                </span>
              </div>
              <div className="metric-value">{losers.length}</div>
              <div className="metric-caption negative-text">
                Demo change below 0%
              </div>
            </article>

            <article className="metric-card">
              <div className="metric-top">
                <span>Watchlist</span>
                <span className="metric-icon amber">
                  <Star size={18} />
                </span>
              </div>
              <div className="metric-value">{watchlist.length}</div>
              <div className="metric-caption">Your saved demo symbols</div>
            </article>
          </section>

          <section className="market-movers-grid" id="market-movers">
            <article className="panel movers-panel">
              <div className="panel-heading">
                <div>
                  <h2>Top gainers</h2>
                  <p>Largest positive demo changes</p>
                </div>
                <span className="metric-icon green">
                  <ArrowUpRight size={18} />
                </span>
              </div>

              {gainers.slice(0, 3).map((stock) => (
                <div className="mover-row" key={stock.symbol}>
                  <div className="stock-monogram">
                    {stock.symbol.slice(0, 2)}
                  </div>
                  <div className="mover-name">
                    <strong>{stock.symbol}</strong>
                    <span>{stock.name}</span>
                  </div>
                  <div className="mover-value">
                    <strong>{formatKES(stock.price)}</strong>
                    <span className="positive-text">
                      +{stock.change.toFixed(2)}%
                    </span>
                  </div>
                </div>
              ))}
            </article>

            <article className="panel movers-panel">
              <div className="panel-heading">
                <div>
                  <h2>Top losers</h2>
                  <p>Largest negative demo changes</p>
                </div>
                <span className="metric-icon red">
                  <ArrowDownRight size={18} />
                </span>
              </div>

              {losers.slice(0, 3).map((stock) => (
                <div className="mover-row" key={stock.symbol}>
                  <div className="stock-monogram">
                    {stock.symbol.slice(0, 2)}
                  </div>
                  <div className="mover-name">
                    <strong>{stock.symbol}</strong>
                    <span>{stock.name}</span>
                  </div>
                  <div className="mover-value">
                    <strong>{formatKES(stock.price)}</strong>
                    <span className="negative-text">
                      {stock.change.toFixed(2)}%
                    </span>
                  </div>
                </div>
              ))}
            </article>
          </section>

          <section className="panel stocks-panel" id="stocks">
            <div className="panel-heading stock-table-heading">
              <div>
                <h2>Listed companies</h2>
                <p>Search, filter and explore the demo stock universe.</p>
              </div>
              <span className="result-count">
                {filteredStocks.length} results
              </span>
            </div>

            <div className="filters">
              <div className="search-box">
                <Search size={18} />
                <input
                  type="search"
                  placeholder="Search name or symbol..."
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                  aria-label="Search companies"
                />
                {search && (
                  <button
                    className="clear-search"
                    onClick={() => setSearch("")}
                    aria-label="Clear search"
                  >
                    <X size={15} />
                  </button>
                )}
              </div>

              <select
                value={sector}
                onChange={(event) => setSector(event.target.value)}
                aria-label="Filter by sector"
              >
                {sectors.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>

              <select
                value={sortBy}
                onChange={(event) => setSortBy(event.target.value)}
                aria-label="Sort companies"
              >
                <option value="name">Sort: Company</option>
                <option value="price">Sort: Price</option>
                <option value="change">Sort: Change</option>
              </select>
            </div>

            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Company</th>
                    <th>Price (KES)</th>
                    <th>Change</th>
                    <th>Volume</th>
                    <th>Demo trend</th>
                    <th>Save</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredStocks.map((stock) => (
                    <tr key={stock.symbol}>
                      <td>
                        <div className="company-cell">
                          <div className="stock-monogram">
                            {stock.symbol.slice(0, 2)}
                          </div>
                          <div className="company-name">
                            <strong>{stock.symbol}</strong>
                            <span>{stock.name}</span>
                            <small>{stock.sector}</small>
                          </div>
                        </div>
                      </td>
                      <td className="price-cell">
                        {formatKES(stock.price)}
                      </td>
                      <td>
                        <span
                          className={`change-pill ${
                            stock.change >= 0 ? "change-up" : "change-down"
                          }`}
                        >
                          {stock.change >= 0 ? "+" : ""}
                          {stock.change.toFixed(2)}%
                        </span>
                      </td>
                      <td className="volume-cell">
                        {formatVolume(stock.volume)}
                      </td>
                      <td>
                        <MiniChart
                          values={stock.history}
                          positive={stock.change >= 0}
                        />
                      </td>
                      <td>
                        <button
                          className={`save-button ${
                            watchlist.includes(stock.symbol) ? "saved" : ""
                          }`}
                          onClick={() => toggleWatchlist(stock.symbol)}
                          aria-label={`${
                            watchlist.includes(stock.symbol)
                              ? "Remove"
                              : "Add"
                          } ${stock.symbol} ${
                            watchlist.includes(stock.symbol)
                              ? "from"
                              : "to"
                          } watchlist`}
                          title="Toggle watchlist"
                        >
                          <Star
                            size={18}
                            fill={
                              watchlist.includes(stock.symbol)
                                ? "currentColor"
                                : "none"
                            }
                          />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {filteredStocks.length === 0 && (
                <div className="empty-state">
                  <Search size={25} />
                  <strong>No companies found</strong>
                  <p>Try another name, symbol or sector.</p>
                  <button
                    className="reset-button"
                    onClick={() => {
                      setSearch("");
                      setSector("All sectors");
                    }}
                  >
                    Reset filters
                  </button>
                </div>
              )}
            </div>

            <div className="table-footer">
              <span>
                Showing {filteredStocks.length} of {stocks.length} demo
                companies
              </span>
              <span>Prices in KES</span>
            </div>
          </section>

          <section className="panel watchlist-panel" id="watchlist">
            <div className="panel-heading">
              <div>
                <h2>My watchlist</h2>
                <p>Companies you have saved during this session.</p>
              </div>
              <Star size={20} className="watchlist-heading-icon" />
            </div>

            {watchlist.length === 0 ? (
              <div className="watchlist-empty">
                <Star size={22} />
                <span>
                  No saved companies yet. Select the star beside a company
                  above to add it here.
                </span>
              </div>
            ) : (
              <div className="watchlist-chips">
                {watchlist.map((symbol) => {
                  const stock = stocks.find(
                    (item) => item.symbol === symbol
                  );

                  if (!stock) return null;

                  return (
                    <div className="watchlist-chip" key={symbol}>
                      <span>
                        <strong>{symbol}</strong>
                        <small>{formatKES(stock.price)}</small>
                      </span>
                      <button
                        className="icon-button"
                        onClick={() => toggleWatchlist(symbol)}
                        aria-label={`Remove ${symbol}`}
                      >
                        <X size={15} />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </section>

          <footer className="page-footer">
            <span>JENGA Market Intelligence · Development preview</span>
            <span>Illustrative data only · Not investment advice</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

