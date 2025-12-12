import customtkinter as ctk
from tkinter import filedialog
import subprocess
import os

# --- Configurações Iniciais ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ViZDoom IA Launcher")
        self.geometry("500x400")
        
        self.config_file_path = "" # Armazena o caminho para o .yaml

        self.grid_columnconfigure(1, weight=1)

        # --- Título ---
        self.title_label = ctk.CTkLabel(self, text="ViZDoom IA Launcher", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10))

        # --- Modo: Host ou Cliente ---
        self.mode_label = ctk.CTkLabel(self, text="Modo de Jogo:")
        self.mode_label.grid(row=1, column=0, padx=20, pady=10, sticky="w")
        self.mode_var = ctk.StringVar(value="Join")
        self.host_switch = ctk.CTkSwitch(self, text="Ser o Host", variable=self.mode_var, onvalue="Host", offvalue="Join", command=self.toggle_ip_entry)
        self.host_switch.grid(row=1, column=1, padx=20, pady=10, sticky="w")

        # --- IP do Host ---
        self.ip_label = ctk.CTkLabel(self, text="IP do Host:")
        self.ip_label.grid(row=2, column=0, padx=20, pady=10, sticky="w")
        self.ip_entry = ctk.CTkEntry(self, placeholder_text="127.0.0.1")
        self.ip_entry.grid(row=2, column=1, padx=20, pady=10, sticky="ew")

        # --- Numero de jogadores ---
        self.jog_label = ctk.CTkLabel(self, text="Numero de jogadores")
        self.jog_label.grid(row=3, column=0, padx=20, pady=10, sticky="w")
        self.jog_entry = ctk.CTkEntry(self, placeholder_text="2")
        self.jog_entry.grid(row=3, column=1, padx=20, pady=10, sticky="ew")

        # --- Porta ---
        self.port_label = ctk.CTkLabel(self, text="Porta:")
        self.port_label.grid(row=4, column=0, padx=20, pady=10, sticky="w")
        self.port_entry = ctk.CTkEntry(self, placeholder_text="5029")
        self.port_entry.insert(0, "5029") # Valor padrão
        self.port_entry.grid(row=4, column=1, padx=20, pady=10, sticky="ew")

        # --- Seleção do Agente (.yaml) ---
        self.config_button = ctk.CTkButton(self, text="Selecionar Agente (.yaml)", command=self.select_config_file)
        self.config_button.grid(row=5, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        self.config_label = ctk.CTkLabel(self, text="Nenhum agente selecionado", text_color="gray")
        self.config_label.grid(row=6, column=0, columnspan=2, padx=20, pady=(0, 10))

        # --- Renderizar (Ver o Jogo) ---
        self.render_var = ctk.BooleanVar(value=True)
        self.render_check = ctk.CTkCheckBox(self, text="Mostrar Janela do Jogo (Render)", variable=self.render_var)
        self.render_check.grid(row=7, column=0, columnspan=2, padx=20, pady=10)

        # --- Botão de Lançar ---
        self.launch_button = ctk.CTkButton(self, text="LANÇAR JOGO", height=40, command=self.launch_game, state="disabled")
        self.launch_button.grid(row=7, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

    def toggle_ip_entry(self):
        """Desabilita o campo de IP se o usuário for o host."""
        if self.mode_var.get() == "Host":
            self.ip_entry.configure(state="disabled", placeholder_text="Você é o Host (IP: 0.0.0.0)")
        else:
            self.ip_entry.configure(state="normal", placeholder_text="127.0.0.1")

    def select_config_file(self):
        """Abre uma caixa de diálogo para selecionar o arquivo .yaml do agente."""
        # Inicia a busca na pasta 'framework' (onde os .yaml devem estar)
        initial_dir = os.path.join(os.getcwd(), "framework")
        file_path = filedialog.askopenfilename(
            title="Selecione o arquivo de configuração do Agente",
            initialdir=initial_dir,
            filetypes=(("YAML files", "*.yaml"), ("All files", "*.*"))
        )
        if file_path:
            self.config_file_path = file_path
            # Mostra o nome do arquivo, não o caminho completo
            filename = os.path.basename(file_path)
            self.config_label.configure(text=f"Agente: {filename}", text_color="green")
            self.launch_button.configure(state="normal") # Ativa o botão de lançar

    def launch_game(self):
        """Constrói e executa o comando final no terminal."""
        
        # 1. Pega os valores da interface
        port = self.port_entry.get()
        mode = self.mode_var.get()
        render = self.render_var.get()
        
        # Garante que o caminho do .yaml seja relativo ao 'framework'
        # Isso é crucial para o python -m
        try:
            # Tenta obter o caminho relativo
            relative_cfg_path = os.path.relpath(self.config_file_path, os.getcwd()).replace("\\", "/")
        except ValueError:
            # Se falhar (ex: discos diferentes), usa o caminho absoluto
            relative_cfg_path = self.config_file_path

        # 2. Constrói o comando base
        # Usamos "python" em vez de "python -m" para simplicidade, 
        # mas precisamos garantir que o CWD esteja correto.
        # Solução melhor: manter o -m
        command = [
            "python", 
            "-m", 
            "framework.client", # Chama o client.py como um módulo
            f"--cfg={relative_cfg_path}"
        ]

        # 3. Adiciona os argumentos
        command.append(f"--port={port}")

        if render:
            command.append("--render")

        if mode == "Host":
            command.append("--host")
            jogadores = self.jog_entry.get()
            command.append(f"--players={jogadores}") # Você pode adicionar um campo para isso
            command.append("--ip=0.0.0.0")
        else:
            ip = self.ip_entry.get()
            if not ip:
                ip = "127.0.0.1" # Padrão se o campo estiver vazio
            command.append(f"--ip={ip}")

        # 4. Executa o comando em um novo processo
        print(f"Executando comando: {' '.join(command)}")
        try:
            # Popen não trava a GUI, ao contrário de 'run'
            subprocess.Popen(command, cwd=os.getcwd()) 
        except Exception as e:
            self.config_label.configure(text=f"Erro ao lançar: {e}", text_color="red")
            return

        # 5. Fecha o launcher
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()