from __future__ import annotations

from typing import Any


class LLMException(Exception):
    """统一异常体系基类。

    每个子类声明：
    - status_code: 对外暴露的 HTTP 语义状态码（即使 CLI 没有监听端口，也保留
      HTTP 语义，便于未来包一层网关直接复用）。
    - retryable:   同一路由内是否值得重试（指数退避）。
    - exit_code:   CLI 语义化退出码。
    """

    status_code: int = 500
    retryable: bool = False
    exit_code: int = 70

    def __init__(self, message: str, *, details: Any = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details
        self.retry_after = retry_after

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": {
                "type": type(self).__name__,
                "message": self.message,
                "status_code": self.status_code,
                "retryable": self.retryable,
            }
        }
        if self.details is not None:
            payload["error"]["details"] = self.details
        if self.retry_after is not None:
            payload["error"]["retry_after"] = self.retry_after
        return payload


class AuthenticationError(LLMException):
    """上游鉴权失败：API Key 无效 / 无权限。"""

    status_code = 401
    retryable = False
    exit_code = 10


class RateLimitError(LLMException):
    """触发上游或本服务限流。"""

    status_code = 429
    retryable = True
    exit_code = 29


class LLMTimeoutError(LLMException):
    """单次 attempt 超时（连接 / 读取）。"""

    status_code = 504
    retryable = True
    exit_code = 54


class ContextLengthExceededError(LLMException):
    """输入超出模型上下文窗口。"""

    status_code = 400
    retryable = False
    exit_code = 13


class UpstreamServerError(LLMException):
    """上游 5xx 或网络中断。"""

    status_code = 502
    retryable = True
    exit_code = 52


class SchemaValidationError(LLMException):
    """结构化输出未通过 JSON Schema 校验（修复循环耗尽后抛出）。"""

    status_code = 422
    retryable = False
    exit_code = 22


class PromptError(LLMException):
    """Prompt 模板缺失 / 渲染失败 / 版本不存在。"""

    status_code = 404
    retryable = False
    exit_code = 60


class ModelNotFoundError(LLMException):
    """请求中的模型别名未在路由表注册。"""

    status_code = 404
    retryable = False
    exit_code = 14


class PromptLeakError(LLMException):
    """输出中检测到系统提示词片段泄漏（Prompt 注入成功的典型痕迹）。"""

    status_code = 403
    retryable = False
    exit_code = 23


class ConfigError(LLMException):
    """配置文件缺失或非法。"""

    status_code = 500
    retryable = False
    exit_code = 61
