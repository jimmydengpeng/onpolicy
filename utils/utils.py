from distutils.spawn import spawn
from enum import Enum
from time import time
from typing import Any, Optional

'''
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)
'''

class Color(Enum):
    # names    = values
    GRAY       = 30
    RED        = 31
    GREEN      = 32
    YELLOW     = 33
    BLUE       = 34
    MAGENTA    = 35
    CYAN       = 36
    WHITE      = 37
    CRIMSON    = 38


def colorize(string: str, color=Color.WHITE, bold=True, highlight=False) -> str:
    attr = []
    # num = color2num[color]
    num = color.value
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

class LogLevel(Enum):
    # enumerated constants 
    SUCCESS = Color.GREEN
    DEBUG = Color.MAGENTA
    INFO = Color.BLUE
    WARNING = Color.YELLOW
    ERROR = Color.RED

LogSymbol = dict({
    # Enum members are hashable & can be used as dictionary keys
    LogLevel.SUCCESS:  "✔", 
    LogLevel.DEBUG:    "",
    LogLevel.INFO:     "",
    LogLevel.WARNING:  "⚠",
    LogLevel.ERROR:    "✖"
})
'''alternatives:  ➤  ☞ ⚑  ◎ ⊙  ⇨ ▶'''

def debug_msg(
        msg: str,
        level=LogLevel.DEBUG,
        color : Optional[Color] = None,
        bold=False,
        inline=False
    ):
    """
    return: symbol msg (same color), e.g.
    ✔ [SUCCESS] SUCCESS
     [DEBUG] DEBUG
     [INFO] INFO
    ⚠ [WARNING] WARNING
    ✖ [ERROR] ERROR
    """
    def colored_prompt(prompt: str) -> str:
        symbol = LogSymbol[level]
        text = symbol + ' [' + prompt + ']'
        return colorize(text, color=level.value, bold=True)

    '''prompt'''
    assert isinstance(level, LogLevel)
    level_name = str(level)[len(LogLevel.__name__)+1:]
    prompt = colored_prompt(level_name)

    '''inline'''
    end = " " if inline else "\n"

    '''Using LogLevel Color'''
    if color == None:
        # print(colorize(prompt, bold=True), colorize(msg, color=level.value, bold=bold))
        print(prompt, colorize(msg, color=level.value, bold=bold), end=end)
    else:
        print(colorize(">>>", bold=True), colorize(msg, color=color, bold=bold), end=end)

def debug_print(
        msg: str,
        args=Any,
        level: LogLevel = LogLevel.DEBUG,
        inline=False
    ):
    debug_msg(msg, level, inline=inline)
    print(args)


def get_formatted_time():
    """
    return: e.g. 20220921_200435
    """
    import time
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def pretty_time(time_in_sec) -> str:
    unit_s = colorize("s", color=Color.GREEN, bold=False)
    unit_m = colorize("m", color=Color.YELLOW, bold=False)
    unit_h = colorize("h", color=Color.RED, bold=False)

    def get_s(t) -> str:
        return colorize(str(t), color=Color.GREEN) + \
               colorize("s", color=Color.GREEN, bold=False)

    def get_m(t) -> str:
        return colorize(str(t), color=Color.YELLOW) + \
               colorize("m", color=Color.YELLOW, bold=False)

    def get_h(t) -> str:
        return colorize(str(t), color=Color.RED) + \
               colorize("h", color=Color.RED, bold=False)

    time_in_sec = int(time_in_sec)
    if time_in_sec <= 100:
        return get_s(time_in_sec)

    elif time_in_sec > 60 and time_in_sec <= (60 * 60):
        m = time_in_sec // 60
        s = time_in_sec % 60
        return get_m(m) + get_s(s)

    elif time_in_sec > (60 * 60): 
        unit = "h"
        h = time_in_sec // (60 * 60)
        remainder = time_in_sec % (60 * 60)
        m = remainder // 60 
        s = remainder % 60
        return get_h(h) + get_m(m) + get_s(s)
    
    else:
        raise NotImplementedError
    
def sec2hms(time_in_sec):
    unit_s = colorize("s", color=Color.GREEN, bold=False)
    unit_m = colorize("m", color=Color.YELLOW, bold=False)
    unit_h = colorize("h", color=Color.RED, bold=False)

    def get_s(t) -> str:
        return str(t) + "s"

    def get_m(t) -> str:
        return str(t) + "m" 

    def get_h(t) -> str:
        return str(t) + "h"

    time_in_sec = int(time_in_sec)
    if time_in_sec <= 100:
        return get_s(time_in_sec)

    elif time_in_sec > 60 and time_in_sec <= (60 * 60):
        m = time_in_sec // 60
        s = time_in_sec % 60
        return get_m(m) + get_s(s)

    elif time_in_sec > (60 * 60): 
        h = time_in_sec // (60 * 60)
        remainder = time_in_sec % (60 * 60)
        m = remainder // 60 
        s = remainder % 60
        return get_h(h) + get_m(m) + get_s(s)
    
def time_str(s):
    """
    Convert seconds to a nicer string showing days, hours, minutes and seconds
    """
    days, remainder = divmod(s, 60 * 60 * 24)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    string = ""
    if days > 0:
        string += f"{int(days):d} days, "
    if hours > 0:
        string += f"{int(hours):d} hours, "
    if minutes > 0:
        string += f"{int(minutes):d} minutes, "
    string += f"{int(seconds):d} seconds"
    return string


''' gym space '''

def get_space_dim(space) -> int:
    import gym.spaces
    if isinstance(space, gym.spaces.Box):
        return space.shape[0]  # type: ignore
    elif isinstance(space, gym.spaces.Discrete):
        return space.n  # type: ignore
    elif isinstance(space, gym.spaces.Dict): # multi-agent
        spaces = [get_space_dim(space[key]) for key in space] # type: ignore
        dim = spaces[0]
        spaces = [s - dim for s in spaces]
        assert not any(spaces)
        return dim
    else:
        raise NotImplementedError


def test_get_space_dim():
    import gym
    from elegantrl.train.config import get_gym_env_args

    env = gym.make("CartPole-v1")
    debug_print("action space:", args=get_space_dim(env.action_space))
    debug_print("obs space:", args=get_space_dim(env.observation_space))
    print(get_gym_env_args(env, if_print=True))

def test_debug_log_functions():
    print("="*10 + " every color " + "="*10)
    for c in Color:
        print(colorize(f"{c}", color=c, bold=False))
        print(colorize(f"{c}.BOLD", color=c))

    print("")
    print("="*10 + " every log level " + "="*10)
    for l in LogLevel:
        level_name = str(l)[len(LogLevel.__name__)+1:]
        debug_msg(level_name, level=l)

    print("")
    print("="*10 + " other color " + "="*10)
    debug_msg("BLUE", color=Color.BLUE)
    debug_msg("CYAN", color=Color.CYAN)
    debug_msg("GREEN", color=Color.GREEN)
    debug_msg("MAGENTA", color=Color.MAGENTA)

    print("")
    print("="*10 + " inline " + "="*10)
    debug_print("hello", args="world", inline=True)

    print("")
    print("="*10 + " newline " + "="*10)
    debug_print("hello", args="world")
    print("")


if __name__ == "__main__":
    # test_debug_log_functions()
    # test_get_space_dim()
    print(formatted_sec(60))
    print(formatted_sec(342))
    print(formatted_sec(12345))