# 📘 DQN 及其變體實作教學文件 (Tutorial)

本文件旨在詳細介紹「Homework 3: DQN and its variants」專案的設計理念、技術實作與實驗結果。

---

## 1. 專案目的 (Purpose)

強化學習 (Reinforcement Learning) 的核心目標是讓代理人 (Agent) 在環境中透過與之互動，學習一套能獲得最大累積獎勵的策略。本專案透過實作深度 Q 網路 (DQN) 及其變體，探討以下核心問題：

*   **基礎 DQN 實作**：理解如何結合神經網路與 Q-Learning，並透過 **經驗回放 (Experience Replay)** 解決資料相關性問題。
*   **消除過度估計**：透過 **Double DQN** 技術，修正標準 DQN 在評估動作價值時容易過度樂觀的問題。
*   **優化網路架構**：透過 **Dueling DQN** 將狀態價值與動作優勢分離，提升在不同狀態下的學習效率。
*   **提升訓練穩定性**：引入 **PyTorch Lightning** 框架，並加入梯度裁剪、優先經驗回放 (PER) 與軟更新 (Polyak Update) 等技術，使模型能適應更複雜的隨機環境。

---

## 2. 實作方法 (Methodology)

### 2.1 環境設計 (GridWorld)
我們設計了一個 3x4 的網格世界，包含：
*   **目標 (G)**：+10 獎勵。
*   **陷阱 (P)**：-10 獎勵。
*   **障礙物 (W)**：無法穿過。
*   **移動懲罰**：每步 -1，鼓勵代理人尋找最短路徑。

環境提供三種模式：`static` (固定起始)、`player` (隨機玩家位置)、`random` (全部隨機)。

### 2.2 技術細節

#### HW3-1: Naive DQN (基礎篇)
*   **核心網路**：簡單的 3 層全連接層 (MLP)。
*   **經驗回放 (Replay Buffer)**：打破資料間的時間相關性，提升隨機梯度下降的穩定性。
*   **策略**：$\epsilon$-greedy 策略，隨訓練進度逐漸減少探索 (Exploration) 並增加開發 (Exploitation)。

#### HW3-2: Double & Dueling DQN (進階篇)
*   **Double DQN**：使用 Online Network 選擇動作，使用 Target Network 評估價值，公式如下：
    $$Y_t = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta_t^-)$$
*   **Dueling DQN**：將 $Q(s, a)$ 拆解為狀態價值 $V(s)$ 與動作優勢 $A(s, a)$，避免在動作對結果影響不大時重複學習。

#### HW3-3: 訓練優化技巧 (穩定篇)
*   **PyTorch Lightning**：結構化程式碼，簡化訓練流程。
*   **優先經驗回放 (PER)**：根據 TD-Error 大小決定取樣機率，優先學習「令人驚訝」的經驗。
*   **梯度裁剪 (Gradient Clipping)**：防止梯度爆炸。
*   **軟更新 (Soft Update)**：目標網路不再是定期複製，而是以 $\tau=0.005$ 的比例緩慢追蹤主網路。

---

## 3. 實驗結果與分析 (Results)

### 3.1 固定模式 (Static Mode)
代理人能在極短時間內（約 100-200 輪）學會最優路徑。`hw3_1_naive_dqn_static.png` 顯示獎勵曲線迅速收斂至正值。

### 3.2 玩家隨機模式 (Player Mode)
在 `hw3_2_dqn_variants_player.png` 中可以觀察到：
*   **Double DQN** 的收斂過程通常比 Naive DQN 更平穩，因為它有效抑制了 Q 值的虛高。
*   **Dueling DQN** 在面對隨機起始位置時展現了更強的泛化能力，其平均獎勵通常略高於標準模型。

### 3.3 全隨機模式 (Random Mode)
這是最具挑戰性的測試。透過 **PyTorch Lightning** 與各類 Tip（如圖 `hw3_3_pl_dqn_random.png`）：
*   模型展現了適應動態環境的能力。
*   **PER** 顯著提升了學習效率，讓代理人在面對極端負獎勵（陷阱）時能更快調整策略。
*   即使目標與障礙位置不斷變換，勝率仍能穩定在一定水準，證明了穩定性技術的價值。

---

## 4. 結語 (Conclusion)

本專案從最基礎的 DQN 出發，逐步演進至結合多項頂尖穩定技術的 DRL 模型。透過實驗，我們證實了：
1.  **經驗回放** 是訓練深層 RL 的基石。
2.  **架構優化** (Dueling) 能加速特徵提取。
3.  **訓練技巧** (PER, Polyak) 則是模型在複雜、不確定環境下生存的關鍵。

---
*詳細程式碼請參閱各 `hw3_x.py` 檔案。*
