# Curious Agent
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=plastic)](CONTRIBUTING.md)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://opensource.org/licenses/mit)
![completion](https://img.shields.io/badge/completion%20state-10%25-blue.svg?style=plastic)

large-scale-curiosity implementation and improvements-

[Paper](https://arxiv.org/abs/1808.04355)<br/>
[Github](https://github.com/openai/large-scale-curiosity)

## Requirements
1. Python 3.7
2. PyTorch
3. Open AI

#### Guidelines

1. Do not use ```print``` function. Instead use
```python
# Create logger object
import logging
logger = logging.getLogger(__name__)

# Log according to the needs.
logger.info("Hello World")
```

2. Implement Train with the "Continuing" Idiomatic Restriction:
```python


continuing = True
state = {}


if continuing:  # the training is being resumed from a halted state
    pass  # i_episode is reloaded as part of the state
else:  # the training is being started from scratch
    self.state.i_episode = 0  # initialize the current number of episodes


# **OK!**: the current episode index can be reset and the algorithm would work seamlessly
while self.state.i_episode < 100000:
    # .
    # .
    # .
    # more training logic
    # .
    # .
    # .
    self.state.i_episode += 1

# **NOT OK!**: the current episode index starts from 0 regardless
for self.state.i_episode in range(0, 100000):
    # .
    # .
    # .
    # more training logic
    # .
    # .
    # .
    pass

```
3. Use the **state** object when implementing the train function to store and retrieve state information
```python
# **OK!**: use the **state** munch object to store and retrieve all state information that the algorithm needs
self.state.i_episode = self.state.i_episode + 1
self.state.eps -= self.state.eps_reduction_rate

# **NOT OK!**: define state variables outside of the **state** munch object.
self.i_episode = self.i_episode + 1
self.eps -= self.eps_reduction_rate
```
