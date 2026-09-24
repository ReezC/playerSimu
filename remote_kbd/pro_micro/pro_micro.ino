// Pro Micro 远程键盘固件
// 通过 USB CDC 串口接收文本指令，模拟成 HID 键盘输出。
// 烧录：Tools->Board->SparkFun Pro Micro, 处理器 ATmega32U4 (3.3V, 8MHz)
//       点「上传」后快速双击 RESET（RST 短接 GND 两次）进入 bootloader。
//
// 指令（每条以换行结束）：
//   PRESS key        按下不放（移动）
//   RELEASE key      松开
//   RELEASEALL       松开所有键 + 三个鼠标按钮（拖拽卡住时的兜底）
//   TAP key ms       点按（按下 -> 等 ms -> 松开）
//   FIX key interval count      固定间隔连按
//   RND key min max count       随机间隔连按
//   HOLD key ms      长按
//   COMBO k1 k2 ms   两键同按
//   MOVE dx dy       鼠标相对移动（自动分块，给大数也对）
//   CLICK btn        鼠标点击（LEFT/RIGHT/MIDDLE，缺省 LEFT）
//   PRESSM btn       按住鼠标按钮
//   RELEASEM btn     松开鼠标按钮
//   SCROLL n         滚轮（正数向上、负数向下）
//   DRAG btn dx dy [steps]  拖拽：按下 -> 分 steps 段移动 -> 松开，**一条指令原子完成**
// 键名：单字符(字母/数字) 或 LEFT/RIGHT/UP/DOWN/SPACE/ENTER/ESC/TAB/CTRL/SHIFT/ALT/F1..F12 等

#include <Keyboard.h>
#include <Mouse.h>

#define MAX_HELD 6
int held[MAX_HELD];
int heldCount = 0;

void setup() {
  Serial.begin(115200);
  Keyboard.begin();
  Mouse.begin();
  randomSeed(analogRead(A0));
  Serial.println("READY");
}

// 键名 -> Keyboard 键码；未知返回 -1
int keyToCode(String k) {
  k.trim();
  if (k.length() == 1) {
    char c = k.charAt(0);
    if (c >= 32 && c <= 126) return (int)c;   // 可打印 ASCII 直接当键码
  }
  k.toUpperCase();
  if (k == "SPACE") return ' ';
  if (k == "ENTER" || k == "RETURN") return KEY_RETURN;
  if (k == "ESC") return KEY_ESC;
  if (k == "TAB") return KEY_TAB;
  if (k == "BACKSPACE") return KEY_BACKSPACE;
  if (k == "UP") return KEY_UP_ARROW;
  if (k == "DOWN") return KEY_DOWN_ARROW;
  if (k == "LEFT") return KEY_LEFT_ARROW;
  if (k == "RIGHT") return KEY_RIGHT_ARROW;
  if (k == "DEL") return KEY_DELETE;
  if (k == "INSERT") return KEY_INSERT;
  if (k == "HOME") return KEY_HOME;
  if (k == "END") return KEY_END;
  if (k == "PGUP") return KEY_PAGE_UP;
  if (k == "PGDN") return KEY_PAGE_DOWN;
  if (k == "GRAVE") return '`';   // 反引号键（` / ~）
  if (k == "CTRL") return KEY_LEFT_CTRL;
  if (k == "SHIFT") return KEY_LEFT_SHIFT;
  if (k == "ALT") return KEY_LEFT_ALT;
  if (k == "GUI" || k == "WIN") return KEY_LEFT_GUI;
  if (k == "RCTRL") return KEY_RIGHT_CTRL;
  if (k == "RSHIFT") return KEY_RIGHT_SHIFT;
  if (k == "RALT") return KEY_RIGHT_ALT;
  if (k == "F1") return KEY_F1;
  if (k == "F2") return KEY_F2;
  if (k == "F3") return KEY_F3;
  if (k == "F4") return KEY_F4;
  if (k == "F5") return KEY_F5;
  if (k == "F6") return KEY_F6;
  if (k == "F7") return KEY_F7;
  if (k == "F8") return KEY_F8;
  if (k == "F9") return KEY_F9;
  if (k == "F10") return KEY_F10;
  if (k == "F11") return KEY_F11;
  if (k == "F12") return KEY_F12;
  return -1;
}

// 按空格拆 token，返回第 i 个（0 起）
String token(String s, int i) {
  int start = 0, idx = 0;
  for (int p = 0; p <= (int)s.length(); p++) {
    if (p == s.length() || s.charAt(p) == ' ') {
      if (idx == i) return s.substring(start, p);
      idx++;
      start = p + 1;
    }
  }
  return "";
}

bool addHeld(int c) {
  for (int i = 0; i < heldCount; i++) if (held[i] == c) return false;
  if (heldCount >= MAX_HELD) return false;
  held[heldCount++] = c;
  return true;
}

bool removeHeld(int c) {
  for (int i = 0; i < heldCount; i++) if (held[i] == c) {
    held[i] = held[--heldCount];
    return true;
  }
  return false;
}

// 鼠标按钮名 -> Mouse 按钮码。CLICK / PRESSM / RELEASEM / DRAG 四处共用一份判定，
// 免得以后加按钮类型时漏掉某一处（那样「按下去的那个」和「松开时那个」就对不上了）。
int mouseBtnCode(String b) {
  b.toUpperCase();
  if (b == "RIGHT") return MOUSE_RIGHT;
  if (b == "MIDDLE") return MOUSE_MIDDLE;
  return MOUSE_LEFT;
}

// 一次相对移动。**必须分块**：Mouse.move() 的入参是 signed char（±127），
// 直接把 4000 传进去会被截成 -96 —— 方向都反了。撞角归零（MOVE -4000 -4000）
// 和长距离拖拽都靠这里才走得对。
#define MOUSE_STEP_MAX 120
void mouseMoveChunked(long dx, long dy) {
  while (dx != 0 || dy != 0) {
    int sx = (int)constrain(dx, -MOUSE_STEP_MAX, MOUSE_STEP_MAX);
    int sy = (int)constrain(dy, -MOUSE_STEP_MAX, MOUSE_STEP_MAX);
    Mouse.move(sx, sy, 0);
    dx -= sx;
    dy -= sy;
  }
}

// 拖拽时每段之间等多久（毫秒）。等一会儿游戏才看得到中间位置 ——
// 瞬移式的拖拽游戏不认。
#define DRAG_STEP_MS 4

//: 拖拽最多分几段。上限是给「别把串口读循环堵太久」用的（40 * 4ms = 160ms）。
#define DRAG_MAX_STEPS 40

void releaseAll() {
  for (int i = 0; i < heldCount; i++) Keyboard.release(held[i]);
  heldCount = 0;
  // 鼠标按钮也要松：拖拽（触控板按住拖 / 界面的「左键」按钮）中途程序崩了、
  // 或触控模式被关掉时，那个按钮会留在按下状态。键盘有 RELEASEALL 兜底，
  // 鼠标原来没有 —— 只能拔板子。固件是最终兜底，清零要放这里。
  Mouse.release(MOUSE_LEFT);
  Mouse.release(MOUSE_RIGHT);
  Mouse.release(MOUSE_MIDDLE);
}

void loop() {
  if (!Serial.available()) return;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (cmd.length() == 0) return;
  String head = token(cmd, 0);
  head.toUpperCase();

  if (head == "PRESS") {
    int code = keyToCode(token(cmd, 1));
    if (code >= 0 && addHeld(code)) Keyboard.press(code);
    Serial.println("DONE");
  }
  else if (head == "RELEASE") {
    int code = keyToCode(token(cmd, 1));
    if (code >= 0 && removeHeld(code)) Keyboard.release(code);
    Serial.println("DONE");
  }
  else if (head == "RELEASEALL") {
    releaseAll();
    Serial.println("DONE");
  }
  else if (head == "MOVE") {
    mouseMoveChunked(token(cmd, 1).toInt(), token(cmd, 2).toInt());
    Serial.println("DONE");
  }
  else if (head == "CLICK") {
    Mouse.click(mouseBtnCode(token(cmd, 1)));
    Serial.println("DONE");
  }
  else if (head == "SCROLL") {
    int n = token(cmd, 1).toInt();
    Mouse.move(0, 0, n);
    Serial.println("DONE");
  }
  else if (head == "PRESSM") {
    Mouse.press(mouseBtnCode(token(cmd, 1)));
    Serial.println("DONE");
  }
  else if (head == "RELEASEM") {
    Mouse.release(mouseBtnCode(token(cmd, 1)));
    Serial.println("DONE");
  }
  else if (head == "DRAG") {
    // 按下 -> 分 steps 段移动 -> 松开，**在固件里一次做完**。
    // 上位机只发这一条，所以中途串口/TLS 断线最多让这次拖拽「没发生」，
    // 不会留下一个按住的左键（那种情况只能拔板子）。
    // 这里 delay() 期间不收新指令（串口缓冲会积压）—— 所以段数有上限
    // （DRAG_MAX_STEPS * DRAG_STEP_MS ≈ 160ms，够用又不至于把读循环堵死）。
    int btn = mouseBtnCode(token(cmd, 1));
    long dx = token(cmd, 2).toInt();
    long dy = token(cmd, 3).toInt();
    int steps = token(cmd, 4).toInt();
    if (steps < 1) steps = 1;
    if (steps > DRAG_MAX_STEPS) steps = DRAG_MAX_STEPS;
    Mouse.press(btn);
    long px = 0, py = 0;
    for (int i = 1; i <= steps; i++) {
      long tx = dx * i / steps;
      long ty = dy * i / steps;
      mouseMoveChunked(tx - px, ty - py);   // 每段自己再按 ±127 分块
      px = tx;
      py = ty;
      delay(DRAG_STEP_MS);
    }
    Mouse.release(btn);
    Serial.println("DONE");
  }
  else if (head == "TAP") {
    int code = keyToCode(token(cmd, 1));
    int ms = token(cmd, 2).toInt();
    if (code >= 0) {
      Keyboard.press(code);
      delay(ms);
      Keyboard.release(code);
    }
    Serial.println("DONE");
  }
  else if (head == "FIX") {
    int code = keyToCode(token(cmd, 1));
    int interval = token(cmd, 2).toInt();
    int count = token(cmd, 3).toInt();
    for (int i = 0; i < count && code >= 0; i++) {
      Keyboard.press(code);
      delay(5);
      Keyboard.release(code);
      if (i < count - 1) delay(interval);
    }
    Serial.println("DONE");
  }
  else if (head == "RND") {
    int code = keyToCode(token(cmd, 1));
    int minMs = token(cmd, 2).toInt();
    int maxMs = token(cmd, 3).toInt();
    int count = token(cmd, 4).toInt();
    for (int i = 0; i < count && code >= 0; i++) {
      Keyboard.press(code);
      delay(5);
      Keyboard.release(code);
      if (i < count - 1) delay(random(minMs, maxMs));
    }
    Serial.println("DONE");
  }
  else if (head == "HOLD") {
    int code = keyToCode(token(cmd, 1));
    int ms = token(cmd, 2).toInt();
    if (code >= 0) {
      Keyboard.press(code);
      delay(ms);
      Keyboard.release(code);
    }
    Serial.println("DONE");
  }
  else if (head == "COMBO") {
    int c1 = keyToCode(token(cmd, 1));
    int c2 = keyToCode(token(cmd, 2));
    int ms = token(cmd, 3).toInt();
    if (c1 >= 0 && c2 >= 0) {
      Keyboard.press(c1);
      Keyboard.press(c2);
      delay(ms);
      Keyboard.release(c1);
      Keyboard.release(c2);
    }
    Serial.println("DONE");
  }
  else {
    Serial.println("ERR");
  }
}
