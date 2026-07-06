---
name: shopping-guide
description: 用户问商品价格、购物清单总价时使用。
---

# 购物助手

1. 从问题里提取商品名和数量（没说数量就按 1）
2. 调用 `get_prices`，传入 items，每项含 name 和 qty
3. 用中文列出明细和合计