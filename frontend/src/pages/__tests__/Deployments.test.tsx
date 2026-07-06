import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { Deployments } from "../Deployments";
import type { DeploymentItem, DeploymentListResponse } from "@/lib/api";

const apiMock = vi.hoisted(() => ({
  listDeployments: vi.fn(),
  createDeployment: vi.fn(),
  startDeployment: vi.fn(),
  stopDeployment: vi.fn(),
  setDeployKillSwitch: vi.fn(),
  getDeploymentEquity: vi.fn(),
  deploymentEventsUrl: vi.fn(() => "/deployments/events"),
}));

vi.mock("@/lib/api", () => ({ api: apiMock }));

vi.mock("@/hooks/useSSE", () => ({
  useSSE: () => ({ connect: vi.fn(), disconnect: vi.fn() }),
}));

vi.mock("@/components/charts/MiniEquityChart", () => ({
  MiniEquityChart: () => <div data-testid="mini-equity" />,
}));

function makeDeployment(overrides: Partial<DeploymentItem> = {}): DeploymentItem {
  return {
    id: "dep1",
    run_id: "runX",
    symbol: "TXFR1.TWF",
    market: "tw_futures",
    environment: "paper",
    interval: "1D",
    sessions: "day",
    allocated_capital: 1_000_000,
    max_order_qty: 5,
    max_daily_orders: 10,
    max_order_notional: 10_000_000,
    enabled: false,
    created_at: "2026-07-06T00:00:00Z",
    last_tick_status: null,
    ...overrides,
  };
}

function makeList(overrides: Partial<DeploymentListResponse> = {}): DeploymentListResponse {
  return {
    deployments: [makeDeployment()],
    kill_switch: false,
    sessions: { paper: { connected: false } },
    ...overrides,
  };
}

function renderPage() {
  return render(
    <MemoryRouter>
      <Deployments />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  apiMock.getDeploymentEquity.mockResolvedValue({ allocated_capital: 1_000_000, points: [] });
});

describe("Deployments", () => {
  it("renders deployment cards with environment badge", async () => {
    apiMock.listDeployments.mockResolvedValue(makeList());
    await act(async () => renderPage());
    expect(await screen.findByText("TXFR1.TWF")).toBeInTheDocument();
    expect(screen.getByText("模擬")).toBeInTheDocument();
    expect(screen.getByText("啟動")).toBeInTheDocument();
  });

  it("live deployments show the danger badge", async () => {
    apiMock.listDeployments.mockResolvedValue(
      makeList({ deployments: [makeDeployment({ environment: "live" })] }),
    );
    await act(async () => renderPage());
    expect(await screen.findByText("正式")).toBeInTheDocument();
  });

  it("start toggle calls the API", async () => {
    apiMock.listDeployments.mockResolvedValue(makeList());
    apiMock.startDeployment.mockResolvedValue({ id: "dep1", enabled: true });
    await act(async () => renderPage());
    await act(async () => {
      fireEvent.click(await screen.findByText("啟動"));
    });
    expect(apiMock.startDeployment).toHaveBeenCalledWith("dep1");
  });

  it("kill switch requires confirmation before engaging", async () => {
    apiMock.listDeployments.mockResolvedValue(makeList());
    apiMock.setDeployKillSwitch.mockResolvedValue({ engaged: true });
    await act(async () => renderPage());
    fireEvent.click(await screen.findByText("全域緊急停止"));
    expect(apiMock.setDeployKillSwitch).not.toHaveBeenCalled();
    await act(async () => {
      fireEvent.click(screen.getByText("確定停止"));
    });
    expect(apiMock.setDeployKillSwitch).toHaveBeenCalledWith(true);
  });

  it("kill switch banner shows when engaged", async () => {
    apiMock.listDeployments.mockResolvedValue(makeList({ kill_switch: true }));
    await act(async () => renderPage());
    expect(
      await screen.findByText(/全域緊急停止已啟動/),
    ).toBeInTheDocument();
  });

  it("empty state prompts creating from Reports", async () => {
    apiMock.listDeployments.mockResolvedValue(makeList({ deployments: [] }));
    await act(async () => renderPage());
    expect(await screen.findByText("還沒有任何部署")).toBeInTheDocument();
  });

  it("create dialog gates live behind typed confirmation field", async () => {
    apiMock.listDeployments.mockResolvedValue(makeList({ deployments: [] }));
    await act(async () => renderPage());
    fireEvent.click(await screen.findByText("新增部署"));
    expect(screen.getByText("模擬倉（simulation）")).toBeInTheDocument();
    // Switch to live -> the typed confirmation field appears.
    fireEvent.click(screen.getByText("正式環境（live，需 CA）"));
    await waitFor(() => {
      expect(screen.getByPlaceholderText("TXFR1.TWF")).toBeInTheDocument();
    });
  });
});
