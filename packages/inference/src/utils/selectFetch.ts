/**
 * Follow-up HTTP calls (queue polling, result downloads, and input URL downloads)
 * use the fetch selected for the inference call. When the caller does not pass one,
 * this is the runtime global fetch.
 *
 * Decoding a `data:` URL stays on global fetch: it is not a network request, and a
 * proxy fetch may not implement the data: scheme.
 */
export function selectFetch(fetchFn?: typeof fetch): typeof fetch {
	return fetchFn ?? fetch;
}
