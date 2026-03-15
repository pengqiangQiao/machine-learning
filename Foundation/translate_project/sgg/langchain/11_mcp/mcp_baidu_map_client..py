# mcp_baidu_map_client.py
import json
import socket

# MCP Server 配置（需和服务端一致）
MCP_SERVER_HOST = "localhost"
MCP_SERVER_PORT = 8080
MCP_VERSION = "1.0"


class MCPClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def call_tool(self, tool_name, params):
        """
        调用MCP Server的指定工具
        :param tool_name: 工具名（如 map_geocode）
        :param params: 工具参数（字典）
        :return: MCP协议响应结果（字典）
        """
        # 构造标准MCP协议请求体
        mcp_request = {
            "mcp": {
                "version": MCP_VERSION,
                "tool": tool_name,
                "params": params
            }
        }

        # 创建Socket连接并发送请求
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            try:
                # 连接MCP Server
                client_socket.connect((self.host, self.port))
                # 发送JSON请求（编码为UTF-8）
                client_socket.send(json.dumps(mcp_request, ensure_ascii=False).encode("utf-8"))
                # 接收响应（最大接收10240字节）
                response_data = client_socket.recv(10240).decode("utf-8")
                # 解析响应
                return json.loads(response_data)
            except ConnectionRefusedError:
                return {
                    "mcp": {"version": MCP_VERSION, "error": {"code": 503, "message": "连接失败：MCP Server未启动"}}
                }
            except json.JSONDecodeError:
                return {
                    "mcp": {"version": MCP_VERSION, "error": {"code": 400, "message": "响应格式错误：非合法JSON"}}
                }
            except Exception as e:
                return {
                    "mcp": {"version": MCP_VERSION, "error": {"code": 500, "message": f"客户端错误：{str(e)}"}}
                }


# 调用示例
if __name__ == "__main__":
    # 创建MCP Client实例
    client = MCPClient(MCP_SERVER_HOST, MCP_SERVER_PORT)

    # 调用地理编码工具
    result = client.call_tool(
        tool_name="map_geocode",
        params={
            "address": "陕西省西安市未来之瞳"
        }
    )

    # 打印结果
    print("MCP Client 调用结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))