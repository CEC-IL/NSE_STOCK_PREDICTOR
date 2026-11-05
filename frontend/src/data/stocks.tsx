
export type Stock = {
  symbol: string;
  name: string;
  sector: string;
  price: number;
  change: number;
  volume: number;
  history: number[];
};

export const stocks: Stock[] = [
  {
    symbol: "SCOM",
    name: "Safaricom PLC",
    sector: "Telecommunications",
    price: 28.4,
    change: 1.25,
    volume: 1250000,
    history: [24, 25, 24.5, 26, 25.4, 27, 26.5, 28.4],
  },
  {
    symbol: "EQTY",
    name: "Equity Group Holdings",
    sector: "Banking",
    price: 51.75,
    change: 0.8,
    volume: 640000,
    history: [48, 49, 47.5, 50, 49.5, 50.2, 51, 51.75],
  },
  {
    symbol: "KCB",
    name: "KCB Group PLC",
    sector: "Banking",
    price: 46.2,
    change: -0.65,
    volume: 420000,
    history: [49, 48.5, 48, 47, 48, 46.8, 46.5, 46.2],
  },
  {
    symbol: "EABL",
    name: "East African Breweries PLC",
    sector: "Manufacturing",
    price: 190.0,
    change: 1.4,
    volume: 85000,
    history: [180, 182, 185, 183, 187, 186, 189, 190],
  },
  {
    symbol: "ABSA",
    name: "Absa Bank Kenya PLC",
    sector: "Banking",
    price: 25.1,
    change: -0.2,
    volume: 210000,
    history: [25.8, 25.6, 25.9, 25.4, 25.5, 25.2, 25.3, 25.1],
  },
  {
    symbol: "COOP",
    name: "Co-operative Bank of Kenya",
    sector: "Banking",
    price: 18.65,
    change: 0.55,
    volume: 310000,
    history: [17.5, 17.8, 17.6, 18, 18.2, 18, 18.4, 18.65],
  },
  {
    symbol: "BAT",
    name: "British American Tobacco Kenya",
    sector: "Manufacturing",
    price: 410.0,
    change: -1.1,
    volume: 12000,
    history: [420, 418, 422, 416, 414, 415, 412, 410],
  },
  {
    symbol: "BAMB",
    name: "Bamburi Cement PLC",
    sector: "Manufacturing",
    price: 54.0,
    change: 2.15,
    volume: 97000,
    history: [49, 50, 48.5, 51, 52, 51.5, 53, 54],
  },
];

export const DATA_MODE = "ILLUSTRATIVE DEMO DATA";
