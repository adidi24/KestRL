""" Utility Funcitons """
import json
import requests

from wandb.sdk.wandb_run import Run

import jax
from jax import lax
from jax import numpy as jnp
from flax import nnx


def set_seed(seed: int = 42):
    return jax.random.PRNGKey(seed)

@nnx.jit
def soft_update(model : nnx.Module, target : nnx.Module, tau: float = 0.05):
    target_state = nnx.state(target)
    online_state = nnx.state(model)
    
    new_state = jax.tree.map(lambda t, o: (1 - tau)*t + tau*o, target_state, online_state)
    nnx.update(target, new_state)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linear interpolation between start_e and end_e over duration steps."""
    slope = (end_e - start_e) / duration
    return jnp.maximum(end_e, start_e + slope * t)


# You have to use Decimal numeral system, not Hexadecimal.
# Use https://htmlcolorcodes.com/color-picker/ and https://www.binaryhexconverter.com/hex-to-decimal-converter
DISCORD_GREEN = 6075785
DISCORD_RED = 9770003
DISCORD_YELLOW = 16763904
DISCORD_BLUE = 3066993

class AlertState:
    STARTED = "started"
    FINISHED = "finished"
    CRASHED = "crashed"
    INFO = "info"
    
def discord_alert(webhook_url: str,
                  run: Run,
                  state: AlertState = AlertState.FINISHED,
                  info: str = None,
                  title: str = None):
    if state == AlertState.FINISHED:
        color = DISCORD_GREEN
        description = "Your Weights & Biases run just finished. Yay!"
        title = f"Triggered: Run finished ({run.name})"
    elif state == AlertState.CRASHED:
        color = DISCORD_RED
        if info is not None:
            description = f"Your Weights & Biases run just crashed. Go check it out. \n\n{info}"
        else:
            description = "Your Weights & Biases run just failed. Go check it out."
        title = f"Triggered: Run failed ({run.name})"
    elif state == AlertState.INFO:
        color = DISCORD_YELLOW
        description = f"Info: {info}"
        title = f"Triggered: Run info ({run.name})"
    elif state == AlertState.STARTED:
        color = DISCORD_BLUE
        description = "Your Weights & Biases run just started. Go check it out."
        title = f"Triggered: Run started ({run.name})"

    headers = {"Content-Type": "application/json"}
    payload = json.dumps({
        "username": "W&B Alerts",
        "avatar_url": "https://avatars.slack-edge.com/2019-03-01/565107977331_b7799dfcbcd352259517_512.png",
        "embeds": [
            {
                "title": title,
                "description": description,
                "url": f"{'' if run is None else run.get_url()}",
                "color": color,
                "fields": [
                    {
                        "name": "User",
                        "value": run.entity,
                        "inline": True,
                    },
                    {
                        "name": "State",
                        "value": state,
                        "inline": True,
                    }
                ]
            }
        ]
    })
    if (run is not None and webhook_url is not None):
        return requests.post(url=webhook_url, data=payload, headers=headers)
    else:
        print("Discord webhook URL is not set. Skipping alert.")
        return None