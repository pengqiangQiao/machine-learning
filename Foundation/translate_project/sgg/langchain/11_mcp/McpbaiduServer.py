# mcp_baidu_map_server.py
import json
import socket
import threading
import requests

# 配置项
BAIDU_MAP_API_KEY = "PPCXRvvKfAZvY9J1ho9ufkaQKGdzE33k"  # 你的百度地图AK
MCP_SERVER_HOST = "localhost"  # 服务端监听地址
MCP_SERVER_PORT = 8080  # 服务端监听端口
MCP_VERSION = "1.0"  # MCP协议版本


class MCPServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def start(self):
        """启动MCP Server"""
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"MCP Server 启动成功：{self.host}:{self.port}")
        print("等待客户端连接...")

        while True:
            client_socket, client_addr = self.server_socket.accept()
            print(f"客户端连接：{client_addr}")
            # 多线程处理客户端请求，支持并发
            threading.Thread(target=self.handle_client, args=(client_socket,)).start()

    def handle_client(self, client_socket):
        """处理单个客户端请求"""
        try:
            # 接收客户端发送的MCP请求（最大接收10240字节）
            request_data = client_socket.recv(10240).decode("utf-8")
            if not request_data:
                return

            # 解析MCP请求体
            request_json = json.loads(request_data)
            mcp_request = request_json.get("mcp", {})
            tool_name = mcp_request.get("tool")
            params = mcp_request.get("params", {})

            # 根据工具名处理请求（仅实现地理编码，可扩展其他工具）
            if tool_name == "map_geocode":
                response = self.handle_geocode(params)
            else:
                response = {
                    "mcp": {
                        "version": MCP_VERSION,
                        "error": {"code": 404, "message": f"工具不存在：{tool_name}"}
                    }
                }

            # 发送MCP响应给客户端
            client_socket.send(json.dumps(response, ensure_ascii=False).encode("utf-8"))

        except json.JSONDecodeError:
            error_response = {
                "mcp": {"version": MCP_VERSION, "error": {"code": 400, "message": "请求格式错误：非合法JSON"}}
            }
            client_socket.send(json.dumps(error_response, ensure_ascii=False).encode("utf-8"))
        except Exception as e:
            error_response = {
                "mcp": {"version": MCP_VERSION, "error": {"code": 500, "message": f"服务端错误：{str(e)}"}}
            }
            client_socket.send(json.dumps(error_response, ensure_ascii=False).encode("utf-8"))
        finally:
            client_socket.close()

    def handle_geocode(self, params):
        """处理地理编码请求（封装百度地图API）"""
        # 校验必填参数
        if "address" not in params:
            return {
                "mcp": {"version": MCP_VERSION, "error": {"code": 400, "message": "参数缺失：address"}}
            }

        # 调用百度地图地理编码API
        api_url = "http://api.map.baidu.com/geocoding/v3/"
        api_params = {
            "address": params["address"],
            "city": params.get("city", ""),
            "output": "json",
            "ak": BAIDU_MAP_API_KEY
        }

        try:
            res = requests.get(api_url, params=api_params, timeout=10)
            res.raise_for_status()
            api_result = res.json()

            if api_result.get("status") == 0:
                result_data = api_result.get("result", {})
                return {
                    "mcp": {
                        "version": MCP_VERSION,
                        "tool": "map_geocode",
                        "result": {
                            "location": result_data.get("location", {}),
                            "formatted_address": result_data.get("formatted_address", ""),
                            "confidence": result_data.get("confidence", 0)
                        }
                    }
                }
            else:
                return {
                    "mcp": {
                        "version": MCP_VERSION,
                        "error": {
                            "code": api_result.get("status", -1),
                            "message": api_result.get("message", "百度地图API调用失败")
                        }
                    }
                }
        except Exception as e:
            return {
                "mcp": {"version": MCP_VERSION, "error": {"code": 503, "message": f"地图API调用失败：{str(e)}"}}
            }


# 启动MCP Server
if __name__ == "__main__":
    server = MCPServer(MCP_SERVER_HOST, MCP_SERVER_PORT)
    server.start()