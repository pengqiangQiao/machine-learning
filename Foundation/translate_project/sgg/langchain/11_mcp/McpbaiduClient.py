import requests
import json

# 1. 配置远程调用核心信息
BAIDU_MAP_API_KEY = "PPCXRvvKfAZvY9J1ho9ufkaQKGdzE33k"
REMOTE_API_ENDPOINT = "http://api.map.baidu.com/geocoding/v3/"
MCP_VERSION = "1.0"

# 2. 封装 MCP 协议的远程调用函数（修复 KeyError 问题）
def call_baidu_mcp_remote(tool_name, params):
    """
    模拟 MCP 协议直接调用百度地图远程 API（无本地 Server）
    """
    # 校验必填参数
    if "address" not in params:
        return {
            "mcp": {
                "version": MCP_VERSION,
                "error": {"code": 400, "message": "参数缺失：address 是必填项"}
            }
        }

    # 构造百度地图原生 API 请求参数
    api_params = {
        "address": params["address"],
        "output": "json",
        "ak": BAIDU_MAP_API_KEY,
        "city": params.get("city", "")
    }

    try:
        response = requests.get(
            url=REMOTE_API_ENDPOINT,
            params=api_params,
            timeout=10
        )
        response.raise_for_status()
        api_result = response.json()

        # 修复核心：用 get 方法取值，避免 KeyError
        if api_result.get("status") == 0:
            result_data = api_result.get("result", {})
            return {
                "mcp": {
                    "version": MCP_VERSION,
                    "tool": tool_name,
                    "result": {
                        "location": result_data.get("location", {}),  # 经纬度（兜底空字典）
                        "formatted_address": result_data.get("formatted_address", ""),  # 兜底空字符串
                        "confidence": result_data.get("confidence", 0)  # 置信度兜底 0
                    }
                }
            }
        else:
            return {
                "mcp": {
                    "version": MCP_VERSION,
                    "error": {
                        "code": api_result.get("status", -1),
                        "message": api_result.get("message", "百度地图 API 调用失败")
                    }
                }
            }
    except requests.exceptions.ConnectionError:
        return {
            "mcp": {
                "version": MCP_VERSION,
                "error": {"code": 503, "message": "远程服务不可用：无法连接百度地图 API"}
            }
        }
    # 捕获 KeyError 等所有异常，明确提示
    except Exception as e:
        return {
            "mcp": {
                "version": MCP_VERSION,
                "error": {"code": 500, "message": f"远程调用失败：{str(e)}，请检查返回字段是否存在"}
            }
        }

# 3. 调用示例
if __name__ == "__main__":
    result = call_baidu_mcp_remote(
        tool_name="map_geocode",
        params={
            "address": "北京市海淀区百度大厦",
            "city": "北京市"
        }
    )
    print("MCP 远程调用结果：")
    print(json.dumps(result, ensure_ascii=False, indent=2))