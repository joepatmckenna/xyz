<script>
	import { MoonRegular, SunRegular } from 'svelte-awesome-icons';
	import { darkMode } from '$lib/stores.js';

	const updateCookie = (cookie) => {
		fetch('/api/cookies', {
			method: 'POST',
			body: JSON.stringify(cookie)
		});
	};

	const toggle = () =>
		darkMode.update((value) => {
			value = !value;
			updateCookie({ darkMode: value });
			return value;
		});
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div on:click={toggle}>
	{#if $darkMode}
		<SunRegular />
	{:else}
		<MoonRegular />
	{/if}
</div>
