#! /bin/bash

sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python-is-python3 \
    tmux \
    htop \
    git \
    nodejs \
    bmon \
    p7zip-full \
    nfs-common \
    libosmesa6-dev \
    patchelf \
    golang


# Python dependencies
cat > $HOME/tpu_requirements.txt <<- EndOfFile
-f https://storage.googleapis.com/jax-releases/libtpu_releases.html
jax[tpu]==0.3.25
tensorflow==2.11.0
flax==0.6.0
optax==0.1.3
distrax==0.1.2
--extra-index-url https://download.pytorch.org/whl/cpu
torch==1.12.1
transformers==4.24.0
datasets==2.7.0
tqdm
ml_collections
wandb==0.13.5
gcsfs==2022.11.0
requests
typing_extensions==4.2.0
protobuf==3.20.2
ipython
cloudpickle==2.0.0
torchvision==0.14.1
timm==0.5.4
einops
h5py
scikit-image==0.19.2
dill
EndOfFile

pip install -r $HOME/tpu_requirements.txt


# VIM configurations
cat > $HOME/.vimrc <<- EndOfFile
set tabstop=4
set shiftwidth=4
set softtabstop=4
set expandtab
set backspace=indent,eol,start
syntax on
EndOfFile

# Tmux configurations
cat > $HOME/.tmux.conf <<- EndOfFile
bind r source-file ~/.tmux.conf \; display-message "█▓░ ~/.tmux.conf reloaded."

# Enable colors, https://github.com/tmux/tmux/wiki/FAQ
set -g default-terminal "tmux-256color"

# start with window 1 (instead of 0)
set -g base-index 1
setw -g pane-base-index 1

set -g prefix C-a

set -g set-titles on
set -g set-titles-string '#(whoami)::#h::#(curl ipecho.net/plain;echo)'

# Status bar customization
set -g status-interval 5
set -g status-left-length 90
set -g status-right-length 60
set -g status-justify left

# send the prefix to client inside window (ala nested sessions)
bind-key a send-prefix

bind-key x kill-pane

# auto reorder
set-option -g renumber-windows on

# default window name
set -g status-left "#[fg=green,bg=colour236] #S "

# default statusbar colors
set-option -g status-style fg=yellow,dim,bg=colour235

# default window title colors
set-window-option -g window-status-style fg=yellow,bg=colour236,dim

# active window title colors
set-window-option -g window-status-current-style fg=brightred,bg=colour236

# basename as window title https://stackoverflow.com/a/37136828
set-window-option -g window-status-current-format '#{window_index} #{pane_current_command} #(echo "#{pane_current_path}" | rev | cut -d'/' -f-3 | rev)'
set-window-option -g window-status-format '#{window_index} #{pane_current_command} #(echo "#{pane_current_path}" | rev | cut -d'/' -f-3 | rev)'

# pane border
set-option -g pane-border-style fg=white #base2
set-option -g pane-active-border-style fg=brightcyan #base1

# enable mouse click
set -g mouse on

# keep window on
set -g remain-on-exit on

# Longer scrollback history
set -g history-limit 50000

# Scroll position indicator
set -g mode-style bg=colour235,fg=colour245

# SSH agent forwarding
# set-environment -g SSH_AUTH_SOCK $SSH_AUTH_SOCK
if-shell '[ -n $SSH_AUTH_SOCK ]' " \
  set-option -sg update-environment \"DISPLAY WINDOWID XAUTHORITY\"; \
  setenv -g SSH_AUTH_SOCK /tmp/ssh_auth_sock_tmux; \
  run-shell \"ln -sf $(find /tmp/ssh-* -type s -readable | head -n 1) /tmp/ssh_auth_sock_tmux\" \
"

# Drag windows on the status bar
bind-key -n MouseDrag1Status swap-window -t=
EndOfFile


# HTop Configurations
mkdir -p $HOME/.config/htop
cat > $HOME/.config/htop/htoprc <<- EndOfFile
# Beware! This file is rewritten by htop when settings are changed in the interface.
# The parser is also very primitive, and not human-friendly.
fields=0 48 17 18 38 39 40 2 46 47 49 1
sort_key=46
sort_direction=1
hide_threads=0
hide_kernel_threads=1
hide_userland_threads=1
shadow_other_users=0
show_thread_names=0
show_program_path=1
highlight_base_name=0
highlight_megabytes=1
highlight_threads=1
tree_view=0
header_margin=1
detailed_cpu_time=0
cpu_count_from_zero=0
update_process_names=0
account_guest_in_cpu_meter=0
color_scheme=0
delay=15
left_meters=CPU Memory Swap
left_meter_modes=1 1 1
right_meters=Tasks LoadAverage Uptime
right_meter_modes=2 2 2
EndOfFile
