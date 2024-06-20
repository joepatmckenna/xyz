---
title: implied movement
date: '2024-06-19'
---

<script>
   import EarningsImpliedMovement from "$lib/assets/media/earnings_implied_movement.png"
</script>

Prior to the earnings call for a highly traded stock, investors expect the stock price to move significantly after the anouncement. The magnitude of the expected move can be estimated from options premiums prior to the call. Let's say we're not sure whether the move will be up or down, but we want to profit from a significant move in either direction. We could buy both at-the-money call and put options at the close of the session before the call (this is called a long straddle strategy). In the case of an ideal move upward, the call would be so far in the money that it could be sold to cover the premium paid for both the call and the put, whose value has crashed. Similarly, in an ideal move downward, the put would gain enough value to close a profit. How much is an ideal move? It's the total premium paid to enter the positions. How often does this work?

<img src={EarningsImpliedMovement} alt="Implied versus actual movement of top stocks" width="200"/>
