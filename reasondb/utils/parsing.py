import json


def get_json_from_response(response: str, start_char="["):
    for r in response.split("```json")[::-1]:
        r = r.split("```")[0]
        r = r.strip()
        if start_char == "[":
            r = "[" + "[".join(r.split("[")[1:])
            r = "]".join(r.split("]")[:-1]) + "]"
        elif start_char == "{":
            r = "{" + "{".join(r.split("{")[1:])
            r = "}".join(r.split("}")[:-1]) + "}"
        else:
            raise ValueError("start_char must be '[' or '{'")
        try:
            json.loads(r)
            return r
        except json.JSONDecodeError:
            continue

    r = response.strip()
    if "```" in response:
        r = r.split("```")[1].strip()
        if r.startswith("json"):
            r = r[len("json") :].strip()

    return r
