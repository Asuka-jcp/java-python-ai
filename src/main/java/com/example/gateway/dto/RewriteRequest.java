package com.example.gateway.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public record RewriteRequest(
        @NotBlank(message = "text 不能为空")
        @Size(min = 50, max = 30000, message = "text 长度需在 50-30000 之间")
        String text
) {
}
