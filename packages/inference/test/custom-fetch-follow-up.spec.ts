import { afterEach, describe, expect, it, vi } from "vitest";
import { InferenceClient } from "../src/InferenceClient.js";
import { inferenceProviderMappingCache } from "../src/lib/getInferenceProviderMapping.js";

/**
 * Regression for huggingface.js#2820: Options.fetch is honored for the model
 * lookup and the initial submit, but queue polling and result downloads used
 * global fetch. data: URL decoding stays on global fetch (it is not a network
 * request, and a proxy fetch may not implement the data: scheme).
 */
const itWithFakeTimers = typeof window !== "undefined" && typeof window.document !== "undefined" ? it.skip : it;

interface RecordedCall {
	via: "custom" | "global";
	url: string;
	authorization: boolean;
}

function jsonResponse(body: unknown): Response {
	return new Response(JSON.stringify(body), {
		headers: { "Content-Type": "application/json" },
	});
}

function mapping(provider: string, model: string, providerId: string, task: string): Response {
	return jsonResponse({
		inferenceProviderMapping: [
			{
				provider,
				hfModelId: model,
				providerId,
				status: "live",
				task,
			},
		],
	});
}

function recordFetch(
	via: "custom" | "global",
	calls: RecordedCall[],
	respond: (url: string) => Response,
): typeof fetch {
	return (async (input: RequestInfo | URL, init?: RequestInit) => {
		const url = input instanceof Request ? input.url : String(input);
		calls.push({
			via,
			url,
			authorization: new Headers(init?.headers).has("Authorization"),
		});
		return respond(url);
	}) as typeof fetch;
}

describe("custom fetch for queue polling and result downloads", () => {
	afterEach(() => {
		vi.unstubAllGlobals();
		vi.useRealTimers();
		inferenceProviderMappingCache.clear();
	});

	itWithFakeTimers("routes fal-ai textToVideo polling, result, and output download through Options.fetch", async () => {
		vi.useFakeTimers();
		const calls: RecordedCall[] = [];
		const respond = (url: string): Response => {
			if (url.includes("/api/models/acme/vid")) {
				return mapping("fal-ai", "acme/vid", "fal-ai/vid", "text-to-video");
			}
			if (url.includes("/status")) {
				return jsonResponse({ status: "COMPLETED" });
			}
			if (url.includes("/requests/r1")) {
				return jsonResponse({ video: { url: "https://cdn.example/out.mp4" } });
			}
			if (url.includes("cdn.example/out.mp4")) {
				return new Response(new Uint8Array([1, 2, 3, 4]), { headers: { "Content-Type": "video/mp4" } });
			}
			if (url.includes("router.huggingface.co/fal-ai/")) {
				return jsonResponse({
					request_id: "r1",
					status: "IN_QUEUE",
					response_url: "https://queue.fal.run/fal-ai/vid/requests/r1",
				});
			}
			throw new Error(`unexpected request ${url}`);
		};
		vi.stubGlobal("fetch", recordFetch("global", calls, respond));
		const customFetch = recordFetch("custom", calls, respond);

		const pending = new InferenceClient("hf_xxx").textToVideo(
			{ model: "acme/vid", provider: "fal-ai", inputs: "a cat" },
			{ fetch: customFetch },
		);
		await vi.advanceTimersByTimeAsync(500);
		const blob = await pending;

		expect(blob).toBeInstanceOf(Blob);
		expect(await blob.arrayBuffer()).toEqual(new Uint8Array([1, 2, 3, 4]).buffer);

		const byVia = (via: RecordedCall["via"]) => calls.filter((call) => call.via === via).map((call) => call.url);
		expect(byVia("global")).toEqual([]);
		expect(byVia("custom")).toEqual([
			"https://huggingface.co/api/models/acme/vid?expand[]=inferenceProviderMapping",
			"https://router.huggingface.co/fal-ai/fal-ai/vid?_subdomain=queue",
			"https://router.huggingface.co/fal-ai/fal-ai/vid/requests/r1/status?_subdomain=queue",
			"https://router.huggingface.co/fal-ai/fal-ai/vid/requests/r1?_subdomain=queue",
			"https://cdn.example/out.mp4",
		]);
		expect(calls.filter((call) => call.url.includes("/requests/r1")).every((call) => call.authorization)).toBe(true);
		expect(calls.find((call) => call.url === "https://cdn.example/out.mp4")?.authorization).toBe(false);
	});

	itWithFakeTimers("keeps using global fetch for fal-ai polling when no custom fetch is set", async () => {
		vi.useFakeTimers();
		const calls: RecordedCall[] = [];
		vi.stubGlobal(
			"fetch",
			recordFetch("global", calls, (url) => {
				if (url.includes("/api/models/acme/vid-default")) {
					return mapping("fal-ai", "acme/vid-default", "fal-ai/vid", "text-to-video");
				}
				if (url.includes("/status")) {
					return jsonResponse({ status: "COMPLETED" });
				}
				if (url.includes("/requests/r1")) {
					return jsonResponse({ video: { url: "https://cdn.example/default.mp4" } });
				}
				if (url.includes("cdn.example/default.mp4")) {
					return new Response(new Uint8Array([9]), { headers: { "Content-Type": "video/mp4" } });
				}
				if (url.includes("router.huggingface.co/fal-ai/")) {
					return jsonResponse({
						request_id: "r1",
						status: "IN_QUEUE",
						response_url: "https://queue.fal.run/fal-ai/vid/requests/r1",
					});
				}
				throw new Error(`unexpected request ${url}`);
			}),
		);

		const pending = new InferenceClient("hf_xxx").textToVideo({
			model: "acme/vid-default",
			provider: "fal-ai",
			inputs: "a cat",
		});
		await vi.advanceTimersByTimeAsync(500);
		await pending;

		expect(calls.every((call) => call.via === "global")).toBe(true);
		expect(calls.map((call) => call.url)).toContain(
			"https://router.huggingface.co/fal-ai/fal-ai/vid/requests/r1/status?_subdomain=queue",
		);
		expect(calls.map((call) => call.url)).toContain("https://cdn.example/default.mp4");
	});

	it("routes replicate textToVideo output downloads through Options.fetch", async () => {
		const calls: RecordedCall[] = [];
		const respond = (url: string): Response => {
			if (url.includes("/api/models/acme/rep")) {
				return mapping("replicate", "acme/rep", "owner/video", "text-to-video");
			}
			if (url.includes("/predictions")) {
				return jsonResponse({ output: "https://cdn.example/replicate.mp4" });
			}
			if (url === "https://cdn.example/replicate.mp4") {
				return new Response(new Uint8Array([7, 7]), { headers: { "Content-Type": "video/mp4" } });
			}
			throw new Error(`unexpected request ${url}`);
		};
		vi.stubGlobal("fetch", recordFetch("global", calls, respond));

		const blob = await new InferenceClient("hf_xxx").textToVideo(
			{ model: "acme/rep", provider: "replicate", inputs: "a cat" },
			{ fetch: recordFetch("custom", calls, respond) },
		);

		expect(new Uint8Array(await blob.arrayBuffer())).toEqual(new Uint8Array([7, 7]));
		expect(calls.filter((call) => call.via === "global")).toEqual([]);
		expect(calls.map((call) => call.url)).toContain("https://cdn.example/replicate.mp4");
	});

	it("routes hf-inference URL downloads through Options.fetch and leaves data: decoding on global fetch", async () => {
		const calls: RecordedCall[] = [];
		const respond = (url: string): Response => {
			if (url.includes("/api/models/acme/img")) {
				return mapping("hf-inference", "acme/img", "acme/img", "text-to-image");
			}
			if (url.includes("/models/acme/img")) {
				return jsonResponse({ output: ["https://cdn.example/hf.png"] });
			}
			if (url === "https://cdn.example/hf.png") {
				return new Response(new Uint8Array([4]), { headers: { "Content-Type": "image/png" } });
			}
			throw new Error(`unexpected request ${url}`);
		};
		vi.stubGlobal("fetch", recordFetch("global", calls, respond));

		const blob = await new InferenceClient("hf_xxx").textToImage(
			{ model: "acme/img", provider: "hf-inference", inputs: "a cat" },
			{ fetch: recordFetch("custom", calls, respond) },
		);
		expect(new Uint8Array(await blob.arrayBuffer())).toEqual(new Uint8Array([4]));
		expect(calls.filter((call) => call.via === "global")).toEqual([]);
		expect(calls.map((call) => call.url)).toContain("https://cdn.example/hf.png");

		calls.length = 0;
		inferenceProviderMappingCache.clear();
		const dataRespond = (url: string): Response => {
			if (url.includes("/api/models/acme/b64")) {
				return mapping("hf-inference", "acme/b64", "acme/b64", "text-to-image");
			}
			if (url.includes("/models/acme/b64")) {
				return jsonResponse({ data: [{ b64_json: "aGVsbG8=" }] });
			}
			if (url.startsWith("data:image/jpeg;base64,")) {
				return new Response(new Uint8Array([8]), { headers: { "Content-Type": "image/jpeg" } });
			}
			throw new Error(`unexpected request ${url}`);
		};
		vi.stubGlobal("fetch", recordFetch("global", calls, dataRespond));
		const decoded = await new InferenceClient("hf_xxx").textToImage(
			{ model: "acme/b64", provider: "hf-inference", inputs: "a cat" },
			{ fetch: recordFetch("custom", calls, dataRespond) },
		);
		expect(new Uint8Array(await decoded.arrayBuffer())).toEqual(new Uint8Array([8]));
		expect(calls.filter((call) => call.via === "custom").every((call) => !call.url.startsWith("data:"))).toBe(true);
		expect(calls.some((call) => call.via === "global" && call.url.startsWith("data:image/jpeg;base64,"))).toBe(true);
	});

	it("routes wavespeed textToVideo polling and output download through Options.fetch", async () => {
		const calls: RecordedCall[] = [];
		const respond = (url: string): Response => {
			if (url.includes("/api/models/acme/wave")) {
				return mapping("wavespeed", "acme/wave", "wavespeed-ai/vid", "text-to-video");
			}
			if (url.endsWith("/api/v3/predictions/job1/result")) {
				return jsonResponse({
					code: 200,
					message: "ok",
					data: {
						status: "completed",
						outputs: ["https://cdn.example/wavespeed.mp4"],
					},
				});
			}
			if (url.includes("/api/v3/wavespeed-ai/vid")) {
				return jsonResponse({
					code: 200,
					message: "ok",
					data: {
						id: "job1",
						urls: { get: "https://api.wavespeed.ai/api/v3/predictions/job1/result" },
					},
				});
			}
			if (url === "https://cdn.example/wavespeed.mp4") {
				return new Response(new Uint8Array([5]), { headers: { "Content-Type": "video/mp4" } });
			}
			throw new Error(`unexpected request ${url}`);
		};
		vi.stubGlobal("fetch", recordFetch("global", calls, respond));

		const blob = await new InferenceClient("hf_xxx").textToVideo(
			{ model: "acme/wave", provider: "wavespeed", inputs: "a cat" },
			{ fetch: recordFetch("custom", calls, respond) },
		);

		expect(new Uint8Array(await blob.arrayBuffer())).toEqual(new Uint8Array([5]));
		expect(calls.filter((call) => call.via === "global")).toEqual([]);
		expect(calls.map((call) => call.url)).toEqual([
			"https://huggingface.co/api/models/acme/wave?expand[]=inferenceProviderMapping",
			"https://router.huggingface.co/wavespeed/api/v3/wavespeed-ai/vid",
			"https://router.huggingface.co/wavespeed/api/v3/predictions/job1/result",
			"https://cdn.example/wavespeed.mp4",
		]);
	});

	it("routes zai-org image input downloads through Options.fetch", async () => {
		const calls: RecordedCall[] = [];
		const respond = (url: string): Response => {
			if (url.includes("/api/models/acme/layout")) {
				return mapping("zai-org", "acme/layout", "glm-ocr", "image-to-text");
			}
			if (url === "https://cdn.example/in.png") {
				return new Response(new Uint8Array([1, 2, 3]), { headers: { "Content-Type": "image/png" } });
			}
			if (url.includes("/layout_parsing")) {
				return jsonResponse({ md_results: "caption" });
			}
			throw new Error(`unexpected request ${url}`);
		};
		vi.stubGlobal("fetch", recordFetch("global", calls, respond));

		// ZAI accepts an image URL at runtime. The public ImageToTextInput type only lists Blob.
		const output = await new InferenceClient("hf_xxx").imageToText(
			{ model: "acme/layout", provider: "zai-org", inputs: "https://cdn.example/in.png" } as never,
			{ fetch: recordFetch("custom", calls, respond) },
		);

		expect(output.generated_text).toBe("caption");
		expect(calls.filter((call) => call.via === "global")).toEqual([]);
		expect(calls.map((call) => call.url)).toContain("https://cdn.example/in.png");
	});
});
