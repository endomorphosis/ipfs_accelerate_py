import os
import sys
import tempfile 
import shutil
import subprocess

class akash_kit:
    def __init__(self, resources, meta=None):
        if meta is None:
            self.meta = {}
            if os.geteuid() == 0:
                self.akash_path = "/usr/local/bin/"
            else:
                self.akash_path = os.path.expanduser("~/.akash/bin/")
                pass
        else:
            self.meta = meta
            if "akash_path" in meta:
                self.akash_path = meta["akash_path"]
            else:
                if os.geteuid() == 0:
                    self.akash_path = "/usr/local/bin/"
                else:
                    self.akash_path = os.path.expanduser("~/.akash/bin/")
                    pass
        if not os.path.exists(self.akash_path):
            os.makedirs(self.akash_path)
        self.env = os.environ.copy()
        os.environ['PATH'] = os.environ['PATH'] + ":" + self.akash_path

    def set_akash_env(self, **kwargs):
        if "key" in kwargs:
            key = kwargs["key"]
        elif "key" in self.meta:
            key = self.meta["key"]
        else:
            key = None
            pass
        if "keyname" in kwargs:
            keyname = kwargs["keyname"]
        elif "keyname" in self.meta:
            keyname = self.meta["keyname"]
        else:
            keyname = None
            pass

        self.env["AKASH_NET"] = "https://raw.githubusercontent.com/akash-network/net/main/mainnet"
        #self.env["AKASH_VERSION"] = "$(curl -s https://api.github.com/repos/akash-network/provider/releases/latest | jq -r '.tag_name')"
        akash_version = subprocess.check_output('curl -s https://api.github.com/repos/akash-network/provider/releases/latest | jq -r ".tag_name"', shell=True).decode('utf-8').replace("\n", "")
        self.env["AKASH_VERSION"] = akash_version       
        #self.env["AKASH_CHAIN_ID"] = "$(curl -s " + self.env["AKASH_NET"] + "/chain-id.txt\")"
        akash_chain_id = subprocess.check_output('curl -s ' + self.env["AKASH_NET"] + '/chain-id.txt', shell=True).decode('utf-8').replace("\n", "")
        self.env["AKASH_CHAIN_ID"] = akash_chain_id
        #self.env["AKASH_NODE"] = "$(curl -s " + self.env["AKASH_NET"] + "/rpc-nodes.txt\")"
        akash_node = subprocess.check_output('curl -s ' + self.env["AKASH_NET"] + '/rpc-nodes.txt', shell=True).decode('utf-8').replace("\n", "")
        self.env["AKASH_NODE"] = akash_node
        self.env["AKASH_GAS"] = "auto"
        self.env["AKASH_GAS_ADJUSTMENT"] = "1.15"
        self.env["AKASH_GAS_PRICES"] = "0.025uakt"
        self.env["AKASH_SIGN_MODE"] = "amino-json"
        self.env["AKASH_KEYRING_BACKEND"] = "os"
        self.env["AKASH_KEY_NAME"] = keyname
        self.env["AKASH_ACCOUNT_ADDRESS"] = key
        return self.env["AKASH_NODE"] + " " + self.env["AKASH_CHAIN_ID"] + " " + self.env["AKASH_KEYRING_BACKEND"]

    def install_akash_cli(self, **kwargs):
        akash_path = ""
        which_unzip = os.system("which unzip")
        if which_unzip != 0:
            os.system("sudo apt-get install unzip")
        which_jq = os.system("which jq")
        if which_jq != 0:
            os.system("sudo apt-get install jq")

        if "akash_path" in kwargs:
            akash_path = kwargs["akash_path"]
        elif self.akash_path is not None:
            akash_path = self.akash_path
        else:
            if os.geteuid() == 0:
                akash_path = "/usr/local/bin/"
            else:
                akash_path = os.path.expanduser("~/.akash/bin/")
                pass

        os.system("cd /tmp && curl -sfL https://raw.githubusercontent.com/akash-network/provider/main/install.sh | bash")
        os.system("mv /tmp/bin/provider-services " + akash_path)

        if os.geteuid() != 0:
           self.env['PATH'] = self.env['PATH'] + ":" + akash_path
        
        return self.test_install_akash_cli()

    def test_install_akash_cli(self, **kwargs):
        try:
            test = subprocess.check_output('which provider-services', shell=True, env=self.env).decode('utf-8').replace("\n", "")
            version = subprocess.check_output('provider-services version', shell=True, env=self.env).decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(e)
            raise e
        finally:
            return True
    
    def set_akash_balance(self, **kwargs):
        balance = subprocess.check_output('provider-services query bank balances --node ' + self.env["AKASH_NODE"] + " " + self.env["AKASH_ACCOUNT_ADDRESS"], shell=True, env=self.env).decode('utf-8')
        return balance
    
    def get_akash_cert(self, **kwargs):
        cert = subprocess.check_output('provider-services tx cert generate client --from ' + self.env["AKASH_KEY_NAME"] , shell=True, env=self.env).decode('utf-8')
        self.env["AKASH_CERT"] = cert
        return cert

    def publish_akash_cert(self, **kwargs):
        try:
            test = subprocess.check_output('provider-services tx publish client --from' + self.env["AKASH_KEY_NAME"], shell=True, env=self.env).decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(e)
            raise e
        finally:
            return True

    def add_akash_key(self, key, **kwargs):
        try:
            test = subprocess.check_output('provider-services keys add ' + key, shell=True, env=self.env).decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(e)
            raise e
        finally:
            return True
        
    def create_akash_deployment(self, deployment, **kwargs):
        try:
            deployment = subprocess.check_output('provider-services tx deployment create ' + deployment ,+ ' --from ' +  self.env['AKASH_KEY_NAME'] , shell=True, env=self.env).decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(e)
            raise e
        finally:
            return deployment

    def view_akash_bids(self, **kwargs):
        try:
            bids = subprocess.check_output('provider-services query market bid list --owner ' + self.env['AKASH_ACCOUNT_ADDRESS'] + ' --node ' + self.env["AKASH_NODE"] + ' --dseq ' + self.env["AKASH_DSEQ"]  + " --state=open" , shell=True, env=self.env).decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(e)
            raise e
        finally:
            return bids
        
    def set_akash_provider(self, provider, **kwargs):
        self.env["AKASH_PROVIDER"] = provider
        return provider

    def create_akash_lease(self, **kwargs):
        try:
            lease = subprocess.check_output('provider-services tx market lease create --dseq ' + self.env["AKASH_DSEQ"] + ' -- provider ' + self.env["akash_provider"] + '--from ' + self.env['AKASH_KEY_NAME'], shell=True, env=self.env).decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(e)
            raise e
        finally:
            return lease
        
    def view_akash_leases(self, **kwargs):
        try:
            leases = subprocess.check_output('provider-services query market lease list --owner ' + self.env['AKASH_ACCOUNT_ADDRESS'] + ' --node ' + self.env["AKASH_NODE"] + ' --dseq ' + self.env["AKASH_DSEQ"] , shell=True, env=self.env).decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(e)
            raise e
        finally:
            return leases

    def close_akash_deployment(self, **kwargs):
        try:
            close = subprocess.check_output('provider-services tx deployment close ' + self.env["AKASH_DSEQ"] + ' --from ' + self.env['AKASH_KEY_NAME'], shell=True, env=self.env).decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(e)
            raise e
        finally:
            return close
        
    def close_all_akash_deployments(self, **kwargs):
        try:
            deployments = subprocess.check_output('provider-services query deployment list --owner ' + self.env['AKASH_ACCOUNT_ADDRESS'] + ' --node ' + self.env["AKASH_NODE"], shell=True, env=self.env).decode('utf-8')
            for deployment in deployments:
                close = subprocess.check_output('provider-services tx deployment close ' + deployment + ' --from ' + self.env['AKASH_KEY_NAME'], shell=True, env=self.env).decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(e)
            raise e
        finally:
            return True
        
    def send_akash_manifest(self, manifest, **kwargs):
        try:
            send = subprocess.check_output('provider-services send-manifest ' + manifest + ' --dseq ' + self.env["AKASH_DSEQ"] + ' --from ' + self.env['AKASH_KEY_NAME'] + '--provider ' + self.env["AKASH_PROVIDER"], shell=True, env=self.env).decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(e)
            raise e
        finally:
            return send
        
    def view_akash_logs(self, **kwargs):
        try:
            logs = subprocess.check_output('provider-services lease logs --dseq ' + self.env["AKASH_DSEQ"] + ' --from ' + self.env['AKASH_KEY_NAME'] + '--provider ' + self.env["AKASH_PROVIDER"], shell=True, env=self.env).decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(e)
            raise e
        finally:
            return logs
        
    def update_akash_manifest(self, manifest, **kwargs):
        try:
            update = subprocess.check_output('provider-services tx deployment update ' + manifest + ' --dseq ' + self.env["AKASH_DSEQ"] + ' --from ' + self.env['AKASH_KEY_NAME'], shell=True, env=self.env).decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(e)
            raise e
        finally:
            pass 
        try:
            send_manifest = subprocess.check_output('provider-services send-manifest ' + manifest + ' --dseq ' + self.env["AKASH_DSEQ"] + ' --from ' + self.env['AKASH_KEY_NAME'] + '--provider ' + self.env["AKASH_PROVIDER"], shell=True, env=self.env).decode('utf-8')
        except subprocess.CalledProcessError as e:
            print(e)
            raise e

if __name__ == "__main__":
    akash = akash_kit({})
    results = akash.set_akash_env()
    if results:
        print("Akash CLI installed successfully")
    else:
        print("Akash CLI installation failed")
    pass